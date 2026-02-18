"""Backend protocols for the ingestkit-image package.

Defines the ``ImageVLMBackend`` protocol for vision-language model backends.
Re-exports ``EmbeddingBackend`` and ``VectorStoreBackend`` from
``ingestkit_core.protocols`` for convenience.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ingestkit_core.protocols import (
    EmbeddingBackend,
    VectorStoreBackend,
)


@runtime_checkable
class ImageVLMBackend(Protocol):
    """Interface for Vision-Language Model backends (e.g., Ollama + llama3.2-vision).

    Accepts raw image bytes and returns a text description.
    """

    def caption(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> str:
        """Generate a text caption for the given image. Returns raw text."""
        ...

    def model_name(self) -> str:
        """Return the VLM model identifier."""
        ...

    def is_available(self) -> bool:
        """Check whether the VLM backend is reachable."""
        ...


__all__ = [
    "ImageVLMBackend",
    "EmbeddingBackend",
    "VectorStoreBackend",
]
