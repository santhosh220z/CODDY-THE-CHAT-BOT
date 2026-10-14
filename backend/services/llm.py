from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional

from backend.utils.config import settings
from backend.utils.logging import logger


class LLMProvider(ABC):
    """Interface the RAG pipeline talks to."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        ...


class OllamaProvider(LLMProvider):
    """Alternative backend using a local Ollama server.

    Kept as a drop-in alternative: llama.cpp decodes single-stream text faster
    than the transformers loop, so if generation feels sluggish on this GPU,
    set LLM_PROVIDER=ollama instead of rewriting anything.
    """

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        # (connect, read). A plain timeout is a cap on the whole response, which
        # truncated generations that stalled longer than that between tokens.
        self._timeout = (10, 300)

    def _payload(self, prompt: str, system_prompt: Optional[str], stream: bool) -> Dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": settings.MODEL_MAX_CONTEXT,
                "temperature": settings.MODEL_TEMPERATURE,
                "top_p": settings.MODEL_TOP_P,
                "num_predict": settings.MAX_NEW_TOKENS,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            response = self._requests.post(
                f"{self.base_url}/api/generate",
                json=self._payload(prompt, system_prompt, False),
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            return f"Error: could not reach Ollama at {self.base_url} ({exc})."

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        import json

        try:
            with self._requests.post(
                f"{self.base_url}/api/generate",
                json=self._payload(prompt, system_prompt, True),
                stream=True,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        yield f"\n[Error: {chunk['error']}]"
                        return
                    token = chunk.get("response", "")
                    if token:
                        yield token
        except Exception as exc:
            logger.error(f"Ollama streaming failed: {exc}")
            yield f"\n[Error: could not reach Ollama at {self.base_url}.]"

    def info(self) -> Dict:
        return {
            "provider": "ollama",
            "model_id": self.model,
            "loaded": True,
            "base_url": self.base_url,
        }


_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Process-wide singleton.

    The transformers provider also keeps the model resident; this only avoids
    constructing the provider twice.
    """
    global _provider
    if _provider is None:
        if settings.LLM_PROVIDER == "ollama":
            _provider = OllamaProvider()
            logger.info(f"LLM provider: Ollama ({settings.OLLAMA_MODEL})")
        else:
            from backend.services.transformers_provider import TransformersProvider

            _provider = TransformersProvider()
            logger.info(
                f"LLM provider: transformers ({settings.HF_MODEL_ID}, "
                f"4-bit={settings.LOAD_IN_4BIT}). Weights load on first request."
            )
    return _provider


def warm_up_in_background() -> None:
    """Start loading the model on a background thread.

    Loading takes roughly 20 seconds. Doing it during startup means the server
    is answering requests immediately while the weights load, instead of the
    first user waiting for it. Failures are logged and ignored: the request
    path will retry and surface the error properly if it is real.
    """
    import threading

    def warm():
        try:
            provider = get_llm_provider()
            if hasattr(provider, "load"):
                logger.info("Warming up the language model in the background...")
                provider.load()
                logger.info("Language model ready.")
        except Exception:
            logger.error("Model warm-up failed; the first request will retry.",
                         exc_info=True)

    thread = threading.Thread(target=warm, name="model-warmup", daemon=True)
    thread.start()
    return thread