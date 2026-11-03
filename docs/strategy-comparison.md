# Transcription strategy comparison

This compares four ways to build a cross-platform Python tool that batch-transcribes `.wav` files of any length:

1. **Ollama + Gemma 4** (what this repo does today)
2. **Hugging Face Transformers + Gemma 4** (same model, run directly in Python)
3. **Whisper** (OpenAI's `openai-whisper`, plus `faster-whisper` and `whisper.cpp`)
4. **Windows `System.Speech` with a custom grammar** (last resort)

Facts were checked in November 2026 against PyPI, the projects' own pages and dependency resolution for Python 3.14
and 3.15. Sources are listed at the end. Accuracy notes are qualitative unless a source is given, because Google
does not publish word error rates for Gemma 4 audio.

## Summary

| | Ollama + Gemma | Hugging Face + Gemma | Whisper family | System.Speech + grammar |
| --- | --- | --- | --- | --- |
| **Accuracy (free speech)** | Good; E4B better than E2B | Same model as Ollama, so about the same; full precision can help a little | Very good, the most benchmarked option; `large-v3`/`turbo` are strong | Poor for dictation; good only for phrases in the grammar |
| **Accuracy (custom terms)** | Good: prompt plus post-correction | Good: prompt, plus fine-tuning possible | Fair: `initial_prompt` hint plus post-correction | Excellent within the grammar, nothing outside it |
| **Speed on CPU** | Moderate (E2B is usable) | Slower than Ollama unless tuned | Fast with `faster-whisper`/`whisper.cpp`; slow with `openai-whisper` | Fast |
| **Speed on GPU** | Fast | Fastest for big batches | Very fast | n/a |
| **Hardware** | Ollama install plus a few GB model; 8 GB+ RAM | PyTorch (about 1 to 3 GB install) plus model; GPU recommended | From tiny (39M params) to large (1.55B); `turbo` needs about 6 GB VRAM | Any Windows PC |
| **OS support** | Windows, macOS, Linux | Windows, macOS, Linux | Windows, macOS, Linux | **Windows only** |
| **Python 3.15** | Yes (this repo needs only NumPy) | Yes (`transformers` 5.22, `torch` 2.15 have 3.15 wheels) | `openai-whisper`: yes. `whisper.cpp` bindings: yes. `faster-whisper`: **not yet** | Only through `pythonnet` or a subprocess |
| **License** | Code MIT, Ollama MIT, Gemma 4 Apache 2.0 | Transformers Apache 2.0, Gemma 4 Apache 2.0 | All MIT (code and weights) | Part of Windows/.NET (MIT for the .NET package) |
| **Maintenance** | Active | Very active (weekly releases) | `openai-whisper` slow; `faster-whisper` and `whisper.cpp` active | Legacy, frozen |
| **Long audio** | Must chunk (30 s model limit); this repo does it | Must chunk (30 s limit) | Built in (30 s windows stitched by the library) | Built in (streams a WAV file) |
| **Executable size** | Small (Ollama installed separately) | Very large (bundles PyTorch) | Large (`openai-whisper`), medium (`faster-whisper`), small (`whisper.cpp`) | Small, but Windows only |

**Recommendation:** keep Ollama + Gemma as the main path. It meets the "same Gemma model" requirement, keeps the
executable small, and already works on Python 3.15 on all three OSes. If accuracy or speed on long recordings
becomes the main problem, a Whisper backend through `whisper.cpp` (or `faster-whisper` once it supports Python 3.15)
is the strongest second option. Hugging Face + Gemma is worth it only for GPU batch jobs or fine-tuning.
`System.Speech` is not a real fallback for free-form speech.

## Is Whisper reaching end-of-life?

Partly true, but not in the way that matters for this project.

**What is true**

- OpenAI is retiring its **hosted API** model `whisper-1` (together with `gpt-4o-transcribe` and
  `gpt-4o-mini-transcribe`). The notice was posted on 26 August 2026, and the shutdown date is
  **26 February 2027**. This only affects the paid cloud API.
- The open-source `openai-whisper` package is maintained slowly. Its latest release is `20250625` (June 2025),
  so there has been no release for about 16 months.
- Its PyPI metadata lists Python 3.8 to 3.13 only, and it ships as a source package only (no wheel). Older versions
  fail to build on new setuptools because they rely on `pkg_resources`. These are signs of thin upkeep.

**What is not accurate**

- **The open-source model and package are not deprecated.** There is no end-of-life notice on the repository,
  and self-hosted Whisper has no shutdown date. The weights are MIT-licensed and will keep working.
- **It does not pin an outdated Python.** `requires-python` is `>=3.8`, which is a minimum, not a cap. The list of
  3.8 to 3.13 is only descriptive metadata. Resolving `openai-whisper==20250625` for **Python 3.15** succeeds on
  Windows, Linux and macOS (arm64), because its dependencies (`torch` 2.15, `numba` 0.68, `tiktoken` 0.14) all ship
  3.15 wheels.

**Where the Python-version risk really is**

- `faster-whisper` 1.2.1 **cannot be installed on Python 3.15 today**, because its dependency `onnxruntime`
  (1.31.1) only has wheels up to 3.14. It works on 3.14. This usually gets fixed within a few months of a Python
  release, but it is a real gap right now.
- `openai-whisper` depends on PyTorch and Numba, which are large and must ship new wheels for each Python release.
  They have for 3.15, but a Whisper tool is only as current as those two projects.
- `whisper.cpp` (through `pywhispercpp` 1.5.1, August 2026) already has 3.15 wheels and no PyTorch dependency.

**Bottom line:** the maintainability concern is fair for the `openai-whisper` package itself, which is slow-moving.
It is not end-of-life, and Python version support is fine today. If Whisper is ever added, use `whisper.cpp` or
`faster-whisper` rather than `openai-whisper`, and never the hosted `whisper-1` API.

## 1. Ollama + Gemma 4 (current approach)

The tool chunks each WAV into about 20-second pieces at quiet moments and sends each piece to a local Ollama server
running `gemma4:e2b` (default) or `gemma4:e4b`.

**Pros**

- Uses the required Gemma model. Gemma 4 is Apache 2.0, so there are no custom license terms.
- Python side is tiny (NumPy only), so the PyInstaller executable stays small and builds easily on every OS.
- Ollama handles GPU or CPU, quantisation and model downloads. Works fully offline after the first pull.
- Because Gemma is a general model, the prompt can carry vocabulary, language and translation instructions.
- Works on Python 3.15 today. Ollama is actively released.

**Cons**

- Users must install and run Ollama separately.
- Gemma 4 audio accepts at most **30 seconds** per request, so long files must be chunked and words near cuts can
  suffer. The repo cuts at the quietest point to reduce this.
- Ollama has no dedicated audio field yet; audio goes through the `images` field. Older Ollama builds do not support
  this, and the API could change.
- An LLM can paraphrase, skip or "tidy up" speech instead of transcribing verbatim, more than a dedicated speech
  model would. No official word error rates are published for Gemma 4 audio.
- Slower than Whisper on CPU for the same accuracy.

## 2. Hugging Face Transformers + Gemma 4

Load `google/gemma-4-E2B-it` or `E4B-it` with `transformers` and PyTorch, and run inference in-process.

**Pros**

- Same Gemma model, so it meets the same-model requirement and gives very similar output.
- No separate server. Full-precision weights, batching several chunks at once on a GPU, and fine-tuning on your own
  recordings (for example with LoRA) are all possible.
- `transformers` 5.22 and `torch` 2.15 support Python 3.15. Both are very actively maintained.

**Cons**

- Same 30-second limit, so you still chunk.
- PyTorch makes installs and executables very large (hundreds of MB to several GB with CUDA). Cross-platform
  PyInstaller builds are fragile.
- On CPU it is usually slower than Ollama, which uses quantised models and llama.cpp kernels.
- You manage devices, dtypes and memory yourself.

## 3. Whisper

A dedicated speech recognition model (MIT license, code and weights). Sizes run from `tiny` (39M parameters) to
`large-v3` (1.55B), with `turbo` (809M) as a fast, near-large option. Three main ways to run it:

| | `openai-whisper` | `faster-whisper` | `whisper.cpp` (`pywhispercpp`) |
| --- | --- | --- | --- |
| Engine | PyTorch | CTranslate2 | C/C++ (ggml) |
| Speed | Baseline | Several times faster, less memory | Fast on CPU, Apple Metal, CUDA, Vulkan |
| Python 3.15 | Yes | **No** (blocked on `onnxruntime`) | Yes |
| Latest release | June 2025 | October 2025 (1.2.1) | August 2026 (`pywhispercpp` 1.5.1) |
| Install size | Large (PyTorch) | Medium | Small |
| Needs ffmpeg | Yes (calls the ffmpeg program to load any file, unless you pass a NumPy array) | No (uses PyAV) | No |

**Pros**

- The most widely benchmarked open speech model; very good accuracy across many languages and accents.
- Built-in long-audio handling: 30-second windows are stitched together, and segment and word timestamps come out
  directly, which suits SRT output.
- Fast CPU options (`faster-whisper`, `whisper.cpp`), and tiny models for weak machines.

**Cons**

- Not Gemma, so it does not meet the same-model requirement if that applies to the whole project.
- Known to hallucinate text on silence or music, and can repeat itself on long files. Voice activity detection
  (VAD) helps.
- Custom vocabulary is only a soft hint (`initial_prompt`, limited to about 224 tokens); no instruction following.
- `openai-whisper` itself is slow-moving (see above). Pick `whisper.cpp` or `faster-whisper` instead.

## 4. Windows System.Speech with a custom grammar (last resort)

The old SAPI-based engine in .NET (`System.Speech.Recognition.SpeechRecognitionEngine`). From Python it can be
reached through `pythonnet`, or by calling a small C# or PowerShell helper as a subprocess.

**Pros**

- Built into Windows; no model download, no GPU, very fast.
- `SetInputToWaveFile` reads a WAV of any length, so long audio needs no chunking.
- With an SRGS or `GrammarBuilder` grammar, recognition of a fixed set of phrases (commands, codes, form answers)
  is very reliable.

**Cons**

- **Windows only.** The `System.Speech` NuGet package for modern .NET throws `PlatformNotSupportedException` on
  macOS and Linux, which breaks the cross-platform requirement.
- Free-form dictation accuracy is far behind Whisper or Gemma, and words outside a custom grammar are not
  recognised at all.
- Legacy technology that Microsoft no longer develops. Expects specific WAV formats (PCM, typical 8 to 16 kHz).
- Not a Python library; needs a .NET bridge.

Use it only for a narrow, Windows-only job where every possible utterance is known in advance.

## Sources

- PyPI JSON metadata for `openai-whisper`, `faster-whisper`, `pywhispercpp`, `transformers`, `torch`, `numba`,
  `tiktoken`, `onnxruntime`, `ctranslate2` (checked 3 November 2026), and `uv pip compile` resolution for
  Python 3.14 and 3.15 on Windows, Linux and macOS.
- openai/whisper releases: <https://github.com/openai/whisper/releases>
- openai/whisper `pyproject.toml`: <https://github.com/openai/whisper/blob/main/pyproject.toml>
- whisper-1 API retirement: <https://www.developersdigest.tech/blog/whisper-1-gpt-4o-transcribe-deprecation>
- Gemma audio capabilities (30 s limit, 16 kHz): <https://ai.google.dev/gemma/docs/capabilities/audio>
- Gemma 4 under Apache 2.0: <https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html>
- System.Speech package: <https://www.nuget.org/packages/System.Speech>
