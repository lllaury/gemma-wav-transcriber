# gemma-wav-transcriber

Batch speech-to-text for `.wav` files using a Gemma model running locally in [Ollama](https://ollama.com).
Point it at a folder of recordings of any length and it writes a transcript next to each one.

- Works on Windows, macOS and Linux
- Runs fully offline once the model is downloaded
- Handles long recordings by splitting them into short chunks at quiet moments, so words aren't cut in half
- Reads 8/16/24/32-bit PCM and 32/64-bit float WAV, mono or stereo, at any sample rate
- Outputs plain text, timestamped text, SRT subtitles or JSON
- Skips files that already have a transcript, so you can re-run it on a growing folder

## Requirements

1. **Ollama** – install from <https://ollama.com/download> and make sure it is running (`ollama serve`, or the desktop app).
2. **An audio-capable Gemma model** – the default is `gemma4:e4b`. The tool downloads it automatically on first run, or you can do it yourself:

   ```sh
   ollama pull gemma4:e4b
   ```

   Only the Gemma 4 **E2B** and **E4B** models understand audio. `gemma4:e4b` is the default for better accuracy; use `--model gemma4:e2b` on machines with less memory.
3. **Python 3.15** (3.12 or newer also works) – only needed if you run from source rather than a prebuilt executable.

## Install

### Prebuilt executable

Download `gemma-transcribe-Windows.exe`, `gemma-transcribe-macOS` or `gemma-transcribe-Linux` from the
[Releases](https://github.com/lllaury/gemma-wav-transcriber/releases) page. On macOS/Linux, make it executable with
`chmod +x gemma-transcribe-*`.

### From source

```sh
git clone https://github.com/lllaury/gemma-wav-transcriber
cd gemma-wav-transcriber
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install .
```

This installs the `gemma-transcribe` command. You can also run it with `python -m gemma_wav_transcriber`.

## Usage

```sh
# Transcribe every .wav in a folder (transcripts are written next to each file)
gemma-transcribe recordings/

# Several files and folders, searching subfolders, writing to one output folder
gemma-transcribe interview.wav recordings/ -r -o transcripts/

# SRT subtitles, or timestamped text
gemma-transcribe recordings/ -f srt
gemma-transcribe recordings/ -t

# Use the smaller model
gemma-transcribe recordings/ -m gemma4:e2b
```

| Option | Description |
| --- | --- |
| `-o, --output-dir DIR` | Write transcripts to `DIR` instead of next to each `.wav` |
| `-m, --model NAME` | Ollama model to use (default `gemma4:e4b`) |
| `-f, --format txt\|srt\|json` | Transcript format (default `txt`) |
| `-t, --timestamps` | Prefix each chunk with `[hh:mm:ss]` in `txt` output |
| `-r, --recursive` | Search folders recursively |
| `--chunk-seconds N` | Target chunk length, 1–30 seconds (default 20) |
| `--host URL` | Ollama server URL (default `http://localhost:11434`) |
| `--num-ctx N` | Model context size (default 8192) |
| `--prompt TEXT` | Replace the transcription instruction, e.g. to add names or jargon |
| `--retries N` | Retries per chunk if the server errors (default 2) |
| `--overwrite` | Re-transcribe files that already have a transcript |
| `--no-pull` | Fail instead of downloading a missing model |

## How it works

1. Each WAV is decoded, mixed down to mono and resampled to 16 kHz.
2. The audio is split into chunks of about 20 seconds. Each cut is placed at the quietest moment in the last few seconds of the chunk, and silent chunks are skipped.
3. Each chunk is sent to the model through Ollama's `/api/chat` endpoint as a base64 WAV, with an instruction to transcribe verbatim.
4. The chunk transcripts are joined in order and written out.

Ollama does not yet have a dedicated audio field in its API, so audio is passed through the `images` field, which
current Ollama versions route to the Gemma 4 audio encoder. If your Ollama version replies with something like
"I can only see an image", update Ollama to the latest release.

## Development

```sh
pip install -e ".[dev]"
pytest
```

To build a standalone executable for your platform:

```sh
pip install ".[build]"
pyinstaller --onefile --name gemma-transcribe --collect-submodules gemma_wav_transcriber src/gemma_wav_transcriber/__main__.py
```

GitHub Actions runs the tests and builds executables for all three platforms on every push. Pushing a tag like `v0.1.0`
attaches the executables to a GitHub release.

## License

[MIT](LICENSE)
