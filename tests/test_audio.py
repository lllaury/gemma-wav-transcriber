import struct
import wave

import numpy as np
import pytest

from gemma_wav_transcriber.audio import (
    TARGET_RATE,
    AudioError,
    load_for_model,
    read_wav,
    resample,
    split_chunks,
    to_wav_bytes,
)


def tone(seconds, rate, freq=440.0, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_pcm16(path, samples, rate, channels=1):
    data = np.repeat(samples[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((data * 32767).astype("<i2").tobytes())


def write_float32(path, samples, rate):
    body = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    riff = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)


def test_read_pcm16_stereo(tmp_path):
    src = tone(1.0, 44100)
    p = tmp_path / "a.wav"
    write_pcm16(p, src, 44100, channels=2)
    samples, rate = read_wav(p)
    assert rate == 44100
    assert samples.shape == src.shape
    assert np.allclose(samples, src, atol=1e-3)


def test_read_float32(tmp_path):
    src = tone(0.5, 22050)
    p = tmp_path / "f.wav"
    write_float32(p, src, 22050)
    samples, rate = read_wav(p)
    assert rate == 22050
    assert np.allclose(samples, src, atol=1e-6)


def test_rejects_non_wav(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(AudioError):
        read_wav(p)


def test_resample_length():
    out = resample(tone(2.0, 48000), 48000)
    assert out.size == 2 * TARGET_RATE


def test_load_for_model(tmp_path):
    p = tmp_path / "a.wav"
    write_pcm16(p, tone(1.5, 8000), 8000)
    assert load_for_model(p).size == int(1.5 * TARGET_RATE)


def test_chunks_cover_audio_and_cut_at_silence():
    rate = TARGET_RATE
    speech = tone(9.0, rate)
    gap = np.zeros(int(0.5 * rate), dtype=np.float32)
    # 9s tone, 0.5s gap, 9s tone, 0.5s gap, 9s tone
    audio = np.concatenate([speech, gap, speech, gap, speech])
    chunks = split_chunks(audio, rate, chunk_seconds=10.0, search_seconds=3.0)
    assert len(chunks) == 3
    # each cut should land inside one of the silent gaps
    assert 9.0 <= chunks[0].end <= 9.5
    assert 18.5 <= chunks[1].end <= 19.0
    assert chunks[-1].end == pytest.approx(audio.size / rate)


def test_silent_chunks_dropped():
    audio = np.concatenate([np.zeros(TARGET_RATE * 25, dtype=np.float32), tone(5.0, TARGET_RATE)])
    chunks = split_chunks(audio, chunk_seconds=20.0)
    assert len(chunks) >= 1
    assert all(c.end > 20.0 for c in chunks)


def test_wav_roundtrip(tmp_path):
    src = tone(1.0, TARGET_RATE)
    p = tmp_path / "r.wav"
    p.write_bytes(to_wav_bytes(src))
    samples, rate = read_wav(p)
    assert rate == TARGET_RATE
    assert np.allclose(samples, src, atol=1e-3)
