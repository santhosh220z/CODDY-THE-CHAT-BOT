"""Local generation via HuggingFace transformers.

Runs entirely on this machine: weights load from the local HF cache and
inference happens on the GPU. Nothing is sent to a hosted API.

The model is loaded once and reused. A 4B model at 4-bit costs ~2.6GB of VRAM
and tens of seconds to load, so reloading it per request is not viable.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from backend.utils.config import settings
from backend.utils.logging import logger


class GenerationError(RuntimeError):
    """The model could not produce a completion."""


class TransformersProvider:
    """LLM provider backed by a locally-loaded transformers model."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = settings.HF_MODEL_ID
        self._lock = threading.Lock()
        self.loaded_at: Optional[float] = None
        self.load_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _quantization_config(self):
        if not settings.LOAD_IN_4BIT:
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(
                torch, settings.BNB_4BIT_COMPUTE_DTYPE, torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )

    def load(self) -> None:
        """Idempotent. Safe to call from several threads."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()

            if not torch.cuda.is_available():
                logger.warning(
                    "No CUDA device found. The model will run on CPU, which is "
                    "far slower. Install a CUDA build of torch if this is unexpected."
                )

            tokenizer = AutoTokenizer.from_pretrained(self._model_id)

            kwargs: Dict[str, Any] = {
                "device_map": "auto" if torch.cuda.is_available() else None,
                "low_cpu_mem_usage": True,
            }
            quantization = self._quantization_config()
            if quantization is not None:
                kwargs["quantization_config"] = quantization
                kwargs["dtype"] = getattr(
                    torch, settings.BNB_4BIT_COMPUTE_DTYPE, torch.float16
                )
            else:
                kwargs["dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(self._model_id, **kwargs)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self.loaded_at = time.time()
            self.load_seconds = time.perf_counter() - started

            device = next(model.parameters()).device
            logger.info(
                f"Loaded {self._model_id} in {self.load_seconds:.1f}s on {device} "
                f"({'4-bit' if settings.LOAD_IN_4BIT else 'full precision'})"
            )
            self._log_memory()

    def _log_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(
                    f"VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved "
                    f"of {total:.2f}GB"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _truncate(self, text: str) -> str:
        """Keep the prompt inside the model's usable context.

        Long documents plus history can exceed the window; trimming from the
        front drops the oldest context first.
        """
        tokenizer = self._tokenizer
        limit = settings.MODEL_MAX_CONTEXT - settings.MAX_NEW_TOKENS - 128
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            return text
        kept = ids[-limit:]
        logger.warning(f"Prompt truncated from {len(ids)} to {limit} tokens.")
        return tokenizer.decode(kept, skip_special_tokens=True)

    def _encode(self, prompt: str, system_prompt: Optional[str]) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._truncate(prompt)})

        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "do_sample": settings.MODEL_TEMPERATURE > 0,
            "temperature": max(settings.MODEL_TEMPERATURE, 1e-5),
            "top_p": settings.MODEL_TOP_P,
            # Guard against the model looping on the same token forever.
            "repetition_penalty": 1.05,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.load()
        import torch

        text = self._encode(prompt, system_prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(**inputs, **self._sampling())

        # Slice off the prompt so only the completion is decoded.
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Yield decoded text incrementally.

        `generate` runs on a worker thread and feeds a queue via
        TextIteratorStreamer, so tokens reach the HTTP client as they are
        produced rather than after the whole answer is finished.
        """
        self.load()
        import torch
        from transformers import TextIteratorStreamer

        text = self._encode(prompt, system_prompt)
        inputs = self._tokenizer(text, return_tensors="pt")

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        kwargs = {**inputs, **self._sampling(), "streamer": streamer}

        error: List[BaseException] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**kwargs)
            except BaseException as exc:  # surfaced on the consumer side
                error.append(exc)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            worker.join(timeout=5)

        if error:
            raise GenerationError(str(error[0])) from error[0]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "provider": "transformers",
            "model_id": self._model_id,
            "loaded": self._model is not None,
            "load_seconds": round(self.load_seconds, 1),
            "quantization": (
                f"{settings.BNB_4BIT_QUANT_TYPE} (compute {settings.BNB_4BIT_COMPUTE_DTYPE})"
                if settings.LOAD_IN_4BIT
                else "none"
            ),
            "max_new_tokens": settings.MAX_NEW_TOKENS,
        }
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info.update(
                    device=props.name,
                    vram_total_gb=round(props.total_memory / (1024**3), 2),
                    vram_allocated_gb=round(torch.cuda.memory_allocated() / (1024**3), 2),
                )
        except Exception:
            pass
        return info