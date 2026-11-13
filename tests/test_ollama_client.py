from gemma_wav_transcriber.ollama_client import OllamaClient, OllamaError


def test_thinking_is_turned_off(monkeypatch):
    sent = []
    client = OllamaClient()

    def chat(payload):
        sent.append(dict(payload))
        return {"message": {"content": " hello ", "thinking": "hidden"}, "done_reason": "stop"}

    monkeypatch.setattr(client, "_chat", chat)
    assert client.transcribe_wav("gemma4:e4b", b"RIFF") == "hello"
    assert sent[0]["think"] is False


def test_think_field_dropped_when_server_rejects_it(monkeypatch):
    sent = []
    client = OllamaClient()

    def chat(payload):
        sent.append(dict(payload))
        if "think" in payload:
            raise OllamaError('Ollama returned HTTP 400: json: unknown field "think"')
        return {"message": {"content": "hello"}}

    monkeypatch.setattr(client, "_chat", chat)
    assert client.transcribe_wav("gemma4:e2b", b"RIFF") == "hello"
    assert client.transcribe_wav("gemma4:e2b", b"RIFF") == "hello"
    assert ["think" in p for p in sent] == [True, False, False]


def test_empty_reply_explains_thinking(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client, "_chat", lambda p: {"message": {"content": "", "thinking": "words"}, "done_reason": "stop"}
    )
    assert client.transcribe_wav("gemma4:e4b", b"RIFF") == ""
    assert "hidden reasoning" in client.empty_reason
