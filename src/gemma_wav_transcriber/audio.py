"""WAV loading, resampling and chunking.

Only numpy is used, so this works the same on Windows, macOS and Linux
without ffmpeg or any native audio library.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_RATE = 16_000

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class AudioError(Exception):
    """Raised when a file cannot be read as WAV audio."""


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # seconds
    end: float  # seconds
    samples: np.ndarray  # float32 mono at TARGET_RATE


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (mono float32 samples in [-1, 1], sample rate).

    Supports 8/16/24/32-bit integer PCM and 32/64-bit float, including
    WAVE_FORMAT_EXTENSIBLE headers.
    """
    data = Path(path).read_bytes()
    if len(data) < 12 or data[0:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise AudioError(f"{path}: not a RIFF/WAVE file")

    fmt = None
    pcm = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body_start = pos + 8
        if chunk_id == b"data" and (size == 0xFFFFFFFF or body_start + size > len(data)):
            # RF64 or truncated/streamed files: take everything that is left.
            size = len(data) - body_start
        body = data[body_start : body_start + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            pcm = body
        pos = body_start + size + (size & 1)

    if fmt is None or pcm is None or len(fmt) < 16:
        raise AudioError(f"{path}: missing fmt or data chunk")

    format_tag, channels, rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 26:
        (format_tag,) = struct.unpack("<H", fmt[24:26])
    if channels < 1 or rate < 1:
        raise AudioError(f"{path}: invalid header")

    width = bits // 8
    usable = len(pcm) - len(pcm) % (width * channels)
    pcm = pcm[:usable]

    if format_tag == _WAVE_FORMAT_PCM:
        if width == 1:
            samples = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
            ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
            samples = ints.astype(np.float32) / 8388608.0
        elif width == 4:
            samples = (np.frombuffer(pcm, dtype="<i4").astype(np.float64) / 2147483648.0).astype(
                np.float32
            )
        else:
            raise AudioError(f"{path}: unsupported PCM bit depth {bits}")
    elif format_tag == _WAVE_FORMAT_IEEE_FLOAT:
        if width == 4:
            samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
        elif width == 8:
            samples = np.frombuffer(pcm, dtype="<f8").astype(np.float32)
        else:
            raise AudioError(f"{path}: unsupported float bit depth {bits}")
    else:
        raise AudioError(f"{path}: unsupported WAV encoding (format tag {format_tag:#06x})")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.clip(samples, -1.0, 1.0).astype(np.float32), rate


def resample(samples: np.ndarray, rate: int, target: int = TARGET_RATE) -> np.ndarray:
    """Resample mono audio to `target` Hz (box low-pass + linear interpolation)."""
    if rate == target or samples.size == 0:
        return samples.astype(np.float32)
    if rate > target:
        # Cheap anti-aliasing: average over the decimation factor first.
        width = int(round(rate / target))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            samples = np.convolve(samples, kernel, mode="same")
    duration = samples.size / rate
    out_len = max(1, int(round(duration * target)))
    src_t = np.arange(samples.size, dtype=np.float64) / rate
    dst_t = np.arange(out_len, dtype=np.float64) / target
    return np.interp(dst_t, src_t, samples).astype(np.float32)


def load_for_model(path: str | Path) -> np.ndarray:
    """Read any supported WAV and return 16 kHz mono float32 samples."""
    samples, rate = read_wav(path)
    return resample(samples, rate)


def _quietest_cut(samples: np.ndarray, lo: int, hi: int, rate: int) -> int:
    """Return the sample index in [lo, hi) with the lowest short-term energy."""
    frame = max(1, rate // 50)  # 20 ms frames
    window = samples[lo:hi]
    n_frames = window.size // frame
    if n_frames < 2:
        return hi
    energy = (window[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * frame + frame // 2


def split_chunks(
    samples: np.ndarray,
    rate: int = TARGET_RATE,
    chunk_seconds: float = 20.0,
    search_seconds: float = 3.0,
    min_seconds: float = 0.3,
    silence_rms: float = 0.002,
) -> list[Chunk]:
    """Split audio into chunks of about `chunk_seconds`.

    Each cut is placed at the quietest point in the last `search_seconds`
    of the chunk so words are not sliced in half. Chunks that are shorter
    than `min_seconds` or effectively silent are dropped.
    """
    target = max(1, int(chunk_seconds * rate))
    search = min(int(search_seconds * rate), target // 2)
    chunks: list[Chunk] = []
    start = 0
    total = samples.size
    while start < total:
        end = start + target
        if end >= total:
            end = total
        elif search > 0:
            end = _quietest_cut(samples, end - search, end, rate)
        piece = samples[start:end]
        if piece.size >= min_seconds * rate and float(np.sqrt(np.mean(piece**2))) > silence_rms:
            chunks.append(Chunk(len(chunks), start / rate, end / rate, piece))
        start = end
    return chunks


def to_wav_bytes(samples: np.ndarray, rate: int = TARGET_RATE) -> bytes:
    """Encode mono float samples as a 16-bit PCM WAV file in memory."""
    ints = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(ints.tobytes())
    return buf.getvalue()
