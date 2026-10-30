# gemma-wav-transcriber

Batch speech-to-text for `.wav` files using a Gemma model running locally in [Ollama](https://ollama.com).
Point it at a folder of recordings of any length and it writes a transcript next to each one.

- Works on Windows, macOS and Linux
- Runs fully offline once the model is downloaded
- Handles long recordings by splitting them into short chunks at quiet moments, so words aren't cut in half
- Reads 8/16/24/32-bit PCM and 32/64-bit float WAV, mono or stereo, at any sample rate
- Outputs plain text, timestamped text, SRT subtitles or JSON
- Skips files that already have a transcript, so you can re-run it on a growing folder
- Custom vocabulary files teach it your domain's terms, how they sound, and your acronyms
- Writes acronyms in capitals, and fixes your listed acronyms even when spelled out, e.g. "f.b.i." becomes "FBI"
- Transcribe in a chosen language, or translate the speech into another one

## Requirements

1. **Ollama** – install from <https://ollama.com/download> and make sure it is running (`ollama serve`, or the desktop app).
2. **An audio-capable Gemma model** – the default is `gemma4:e2b`. The tool downloads it automatically on first run, or you can do it yourself:

   ```sh
   ollama pull gemma4:e2b
   ```

   Only the Gemma 4 **E2B** and **E4B** models understand audio. `gemma4:e2b` is the default because it is fast enough to run on a CPU; use `--model gemma4:e4b` for somewhat better accuracy if you have a GPU or plenty of time.
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

# Use the larger, more accurate model
gemma-transcribe recordings/ -m gemma4:e4b

# Domain vocabulary, Spanish audio, and translation to English
gemma-transcribe recordings/ --vocab medical.txt
gemma-transcribe recordings/ -l es
gemma-transcribe recordings/ -l es --translate-to en
```

| Option | Description |
| --- | --- |
| `-o, --output-dir DIR` | Write transcripts to `DIR` instead of next to each `.wav` |
| `-m, --model NAME` | Ollama model to use (default `gemma4:e2b`) |
| `-f, --format txt\|srt\|json` | Transcript format (default `txt`) |
| `-t, --timestamps` | Prefix each chunk with `[hh:mm:ss]` in `txt` output |
| `-r, --recursive` | Search folders recursively |
| `--chunk-seconds N` | Target chunk length, 1–30 seconds (default 20) |
| `--host URL` | Ollama server URL (default `http://localhost:11434`) |
| `--num-ctx N` | Model context size (default 8192) |
| `-v, --vocab FILE` | Custom vocabulary file (see below). Can be given more than once |
| `--no-correct` | Give the vocabulary to the model but don't post-correct the transcript |
| `-l, --language LANG` | Language spoken in the audio, as a code (`es`) or name (`Spanish`). Default: auto-detect |
| `--translate-to LANG` | Output a translation into `LANG` instead of a verbatim transcript |
| `--no-acronym-rule` | Don't tell the model to write acronyms in capitals |
| `--prompt TEXT` | Replace the base transcription instruction |
| `--show-prompt` | Print the full instruction sent to the model |
| `--retries N` | Retries per chunk if the server errors (default 2) |
| `--overwrite` | Re-transcribe files that already have a transcript |
| `--no-pull` | Fail instead of downloading a missing model |

## Custom vocabulary

Speech models often mishear specialist words: product names, jargon, people's names, acronyms. A vocabulary file
lists them, along with how they sound if the model tends to get them wrong. It is plain UTF-8 text, one entry per line
(see [`examples/vocabulary.txt`](examples/vocabulary.txt)):

```ini
# Lines starting with # are comments ("C#" is still a term).
[terms]
Kubernetes                      # spell this term exactly like this
cooper netties = Kubernetes     # how it sounds = how to write it
post gres, postgres = PostgreSQL  # several ways it may be heard

[acronyms]
API, GPU, NASA                  # always written in capitals
hip ah, hippa = HIPAA           # an acronym said as a word
```

The vocabulary is used in two ways:

1. **As context for the model.** The terms, acronyms and pronunciation guide are added to the instruction sent with every
   chunk, so the model knows which spellings to expect.
2. **As post-correction.** After each chunk is transcribed, anything matching a "sounds like" phrase is replaced with its
   written form (ignoring case, and treating spaces and hyphens alike). Terms get their exact spelling back
   (`kubernetes` becomes `Kubernetes`). Acronyms are fixed to capitals, including
   spelled-out forms such as `Nasa`, `n a s a` or `N.A.S.A.`. Use `--no-correct` to turn this off and rely on the
   model alone.

Acronyms that are also everyday words (`IT`, `US`, `AM`, ...) are only corrected when they are spelled out letter by
letter or already in capitals, so "it" is never turned into "IT". Even without a vocabulary file, the model is told to
write acronyms in capitals without periods; `--no-acronym-rule` turns that off.

Keep "sounds like" phrases specific: a phrase that is also ordinary speech will be replaced everywhere it appears.

## Languages

By default the model transcribes whatever language is spoken. If you know the language, `-l/--language` tells the
model, which helps with accents, short clips and mixed-language recordings. It accepts a two-letter code (`en`, `es`,
`fr`, `de`, `ja`, `zh`, ...) or a language name. `--translate-to` asks for a translation instead of a transcript, for
example `-l es --translate-to en` turns Spanish speech into English text. Gemma 4 supports speech in many languages,
but accuracy is best for widely spoken ones.

## How it works

1. Each WAV is decoded, mixed down to mono and resampled to 16 kHz.
2. The audio is split into chunks of about 20 seconds. Each cut is placed at the quietest moment in the last few seconds of the chunk, and silent chunks are skipped.
3. Each chunk is sent to the model through Ollama's `/api/chat` endpoint as a base64 WAV, with an instruction to transcribe
   verbatim (plus the language and vocabulary, if given).
4. Each chunk's text is post-corrected with the vocabulary, then the chunks are joined in order and written out.

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
