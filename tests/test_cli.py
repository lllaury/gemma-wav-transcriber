import json

from gemma_wav_transcriber import cli

from .test_audio import tone, write_pcm16


class FakeClient:
    calls = 0

    def __init__(self, host):
        pass

    def has_model(self, model):
        return True

    def transcribe_wav(self, model, wav_bytes, prompt, num_ctx):
        assert wav_bytes[:4] == b"RIFF"
        FakeClient.calls += 1
        return f"segment {FakeClient.calls}"


def test_batch_writes_transcripts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    FakeClient.calls = 0
    write_pcm16(tmp_path / "short.wav", tone(3.0, 16000), 16000)
    write_pcm16(tmp_path / "long.wav", tone(45.0, 44100), 44100)
    out = tmp_path / "out"

    assert cli.main([str(tmp_path), "-o", str(out)]) == 0
    assert (out / "short.txt").read_text().strip().startswith("segment")
    assert (out / "long.txt").read_text().count("segment") == 3

    # second run skips existing transcripts
    calls = FakeClient.calls
    assert cli.main([str(tmp_path), "-o", str(out)]) == 0
    assert FakeClient.calls == calls


def test_srt_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    FakeClient.calls = 0
    write_pcm16(tmp_path / "a.wav", tone(25.0, 16000), 16000)
    assert cli.main([str(tmp_path / "a.wav"), "-f", "srt"]) == 0
    srt = (tmp_path / "a.srt").read_text()
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert cli.main([str(tmp_path / "a.wav"), "-f", "json"]) == 0
    data = json.loads((tmp_path / "a.json").read_text())
    assert data[0]["start"] == 0.0 and data[-1]["end"] == 25.0


def test_bad_file_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)
    (tmp_path / "bad.wav").write_bytes(b"garbage")
    assert cli.main([str(tmp_path)]) == 1


def test_vocab_language_and_correction(tmp_path, monkeypatch):
    prompts = []

    class Client(FakeClient):
        def transcribe_wav(self, model, wav_bytes, prompt, num_ctx):
            prompts.append(prompt)
            return "we use cooper netties at n.a.s.a."

    monkeypatch.setattr(cli, "OllamaClient", Client)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("cooper netties = Kubernetes\n[acronyms]\nNASA\n", encoding="utf-8")
    write_pcm16(tmp_path / "a.wav", tone(3.0, 16000), 16000)

    assert cli.main([str(tmp_path / "a.wav"), "--vocab", str(vocab), "-l", "en"]) == 0
    assert (tmp_path / "a.txt").read_text().strip() == "we use Kubernetes at NASA."
    assert "The speech is in English" in prompts[0] and "Kubernetes" in prompts[0]

    assert cli.main([str(tmp_path / "a.wav"), "-v", str(vocab), "--no-correct", "--overwrite"]) == 0
    assert (tmp_path / "a.txt").read_text().strip() == "we use cooper netties at n.a.s.a."


def test_bad_vocab_file(tmp_path, capsys):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[wrong]\n", encoding="utf-8")
    assert cli.main([str(tmp_path), "--vocab", str(vocab)]) == 2
    assert "unknown section" in capsys.readouterr().err


def test_phonetic_spelling(tmp_path, monkeypatch):
    class Client(FakeClient):
        def transcribe_wav(self, model, wav_bytes, prompt, num_ctx):
            return "our code is hotel india papa alpha alpha"

    monkeypatch.setattr(cli, "OllamaClient", Client)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[acronyms]\nHIPAA\n", encoding="utf-8")
    write_pcm16(tmp_path / "a.wav", tone(3.0, 16000), 16000)
    assert cli.main([str(tmp_path / "a.wav"), "-v", str(vocab)]) == 0
    assert (tmp_path / "a.txt").read_text().strip() == "our code is HIPAA"
