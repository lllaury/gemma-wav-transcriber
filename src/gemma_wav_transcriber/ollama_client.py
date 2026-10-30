"""Minimal Ollama HTTP client using only the standard library."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Callable

from .prompt import DEFAULT_PROMPT

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 600.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise OllamaError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(
                f"Could not reach Ollama at {self.host} ({e}). Is `ollama serve` running?"
            ) from e

    def list_models(self) -> list[str]:
        with self._request("GET", "/api/tags") as resp:
            body = json.load(resp)
        return [m.get("name", "") for m in body.get("models", [])]

    def has_model(self, model: str) -> bool:
        names = self.list_models()
        wanted = model if ":" in model else model + ":latest"
        return wanted in names or model in names

    def pull(self, model: str, on_status: Callable[[str], None] | None = None) -> None:
        with self._request("POST", "/api/pull", {"model": model, "stream": True}) as resp:
            for line in resp:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(msg["error"])
                if on_status and msg.get("status"):
                    on_status(msg["status"])

    def transcribe_wav(
        self,
        model: str,
        wav_bytes: bytes,
        prompt: str = DEFAULT_PROMPT,
        num_ctx: int = 8192,
        temperature: float = 0.0,
    ) -> str:
        """Send one WAV clip to the model and return its transcript.

        Ollama has no dedicated audio field yet; audio-capable models such as
        Gemma 4 E2B/E4B accept a base64 WAV passed through `images`.
        """
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(wav_bytes).decode("ascii")],
                }
            ],
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        with self._request("POST", "/api/chat", payload) as resp:
            body = json.load(resp)
        if "error" in body:
            raise OllamaError(body["error"])
        return body.get("message", {}).get("content", "").strip()
