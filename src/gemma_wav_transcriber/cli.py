"""Command-line interface: batch-transcribe .wav files with Gemma via Ollama."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .audio import TARGET_RATE, AudioError, Chunk, load_for_model, split_chunks, to_wav_bytes
from .ollama_client import DEFAULT_HOST, OllamaClient, OllamaError
from .phonetic import to_letters
from .prompt import DEFAULT_PROMPT, build_prompt, language_name
from .vocab import Vocabulary, VocabError, load as load_vocab

DEFAULT_MODEL = "gemma4:e2b"
FORMATS = ("txt", "srt", "json")


def find_wavs(inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() == ".wav")
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: {item} not found, skipping", file=sys.stderr)
    seen: set[Path] = set()
    unique = []
    for f in sorted(files):
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _ts(seconds: float, sep: str = ".") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render(segments: list[tuple[Chunk, str]], fmt: str, timestamps: bool) -> str:
    segments = [(c, t) for c, t in segments if t]
    if fmt == "srt":
        blocks = [
            f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n{t}\n"
            for i, (c, t) in enumerate(segments, 1)
        ]
        return "\n".join(blocks)
    if fmt == "json":
        return json.dumps(
            [{"start": round(c.start, 3), "end": round(c.end, 3), "text": t} for c, t in segments],
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    if timestamps:
        return "\n".join(f"[{_ts(c.start)[:8]}] {t}" for c, t in segments) + "\n"
    return " ".join(t for _, t in segments) + "\n"


def output_path(wav: Path, out_dir: Path | None, fmt: str) -> Path:
    base = out_dir if out_dir else wav.parent
    return base / (wav.stem + "." + fmt)


def transcribe_file(
    client: OllamaClient, wav: Path, args: argparse.Namespace, prompt: str, vocab: Vocabulary
) -> list[tuple[Chunk, str]]:
    samples = load_for_model(wav)
    chunks = split_chunks(samples, TARGET_RATE, chunk_seconds=args.chunk_seconds)
    duration = samples.size / TARGET_RATE
    print(f"  {duration:.1f}s of audio, {len(chunks)} chunk(s)")
    results = []
    for chunk in chunks:
        t0 = time.monotonic()
        text = ""
        for attempt in range(1, args.retries + 2):
            try:
                text = client.transcribe_wav(
                    args.model, to_wav_bytes(chunk.samples), prompt=prompt, num_ctx=args.num_ctx
                )
                break
            except OllamaError as e:
                if attempt > args.retries:
                    raise
                print(f"    chunk {chunk.index + 1}: {e}; retrying", file=sys.stderr)
                time.sleep(2 * attempt)
        print(
            f"    chunk {chunk.index + 1}/{len(chunks)} "
            f"[{_ts(chunk.start)[:8]}-{_ts(chunk.end)[:8]}] {time.monotonic() - t0:.1f}s"
        )
        if not args.no_phonetic:
            text = to_letters(text)
        if vocab and not args.no_correct:
            text = vocab.correct(text)
        results.append((chunk, text))
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gemma-transcribe",
        description="Batch-transcribe .wav files using a Gemma model running in Ollama.",
    )
    p.add_argument("inputs", nargs="+", help=".wav files and/or folders containing .wav files")
    p.add_argument("-o", "--output-dir", type=Path, help="write transcripts here (default: next to each .wav)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("-f", "--format", choices=FORMATS, default="txt", help="transcript format (default: txt)")
    p.add_argument("-t", "--timestamps", action="store_true", help="prefix each chunk with a timestamp (txt only)")
    p.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
    p.add_argument("--chunk-seconds", type=float, default=20.0, help="target chunk length in seconds (default: 20)")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server URL (default: {DEFAULT_HOST})")
    p.add_argument("--num-ctx", type=int, default=8192, help="model context size (default: 8192)")
    p.add_argument("-l", "--language", help="language spoken in the audio, e.g. en, es, French (default: auto-detect)")
    p.add_argument("--translate-to", metavar="LANG", help="translate the speech into this language instead of transcribing it")
    p.add_argument("-v", "--vocab", action="append", type=Path, default=[], metavar="FILE",
                   help="custom vocabulary file of terms, pronunciations and acronyms (can be repeated)")
    p.add_argument("--no-correct", action="store_true",
                   help="only give the vocabulary to the model; do not post-correct the transcript")
    p.add_argument("--no-acronym-rule", action="store_true",
                   help="do not tell the model to write acronyms in capitals")
    p.add_argument("--no-phonetic", action="store_true",
                   help='do not turn NATO phonetic spelling ("alpha bravo charlie") into letters')
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="override the transcription instruction")
    p.add_argument("--show-prompt", action="store_true", help="print the full instruction sent to the model")
    p.add_argument("--retries", type=int, default=2, help="retries per chunk on server errors (default: 2)")
    p.add_argument("--overwrite", action="store_true", help="re-transcribe files that already have a transcript")
    p.add_argument("--no-pull", action="store_true", help="do not download the model if it is missing")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1.0 <= args.chunk_seconds <= 30.0:
        print("error: --chunk-seconds must be between 1 and 30 (Gemma handles about 30s per clip)", file=sys.stderr)
        return 2

    vocab = Vocabulary()
    try:
        for path in args.vocab:
            vocab.merge(load_vocab(path.expanduser()))
    except VocabError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    prompt = build_prompt(
        args.prompt, args.language, args.translate_to, vocab,
        acronyms=not args.no_acronym_rule, phonetic=not args.no_phonetic,
    )
    if vocab:
        print(
            f"Vocabulary: {len(vocab.terms)} term(s), {len(vocab.acronyms)} acronym(s), "
            f"{len(vocab.sounds_like)} pronunciation(s)"
        )
    if args.language or args.translate_to:
        heard = language_name(args.language) or "auto-detect"
        print(f"Language: {heard}" + (f", translating to {language_name(args.translate_to)}" if args.translate_to else ""))
    if args.show_prompt:
        print(f"Prompt: {prompt}")

    wavs = find_wavs(args.inputs, args.recursive)
    if not wavs:
        print("No .wav files found.", file=sys.stderr)
        return 1

    client = OllamaClient(args.host)
    try:
        if not client.has_model(args.model):
            if args.no_pull:
                print(f"error: model {args.model} is not installed (run `ollama pull {args.model}`)", file=sys.stderr)
                return 1
            print(f"Downloading {args.model} (first run only)...")
            last = ""
            def status(s: str) -> None:
                nonlocal last
                if s != last:
                    print(f"  {s}")
                    last = s
            client.pull(args.model, status)
    except OllamaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for n, wav in enumerate(wavs, 1):
        out = output_path(wav, args.output_dir, args.format)
        if out.exists() and not args.overwrite:
            print(f"[{n}/{len(wavs)}] {wav.name}: transcript exists, skipping")
            skipped += 1
            continue
        print(f"[{n}/{len(wavs)}] {wav}")
        try:
            segments = transcribe_file(client, wav, args, prompt, vocab)
        except (AudioError, OllamaError, OSError) as e:
            print(f"  failed: {e}", file=sys.stderr)
            failed += 1
            continue
        out.write_text(render(segments, args.format, args.timestamps), encoding="utf-8")
        print(f"  -> {out}")
        done += 1

    print(f"\nDone: {done} transcribed, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
