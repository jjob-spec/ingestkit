"""Backend protocols for the ingestkit-image package.

Defines the ``ImageVLMBackend`` protocol for vision-language model backends
and ``ImageOCRBackend`` for OCR text extraction backends.
Re-exports ``EmbeddingBackend`` and ``VectorStoreBackend`` from
``ingestkit_core.protocols`` for convenience.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ingestkit_core.protocols import (
    EmbeddingBackend,
    VectorStoreBackend,
)


# ---------------------------------------------------------------------------
# OCR Result Model
# ---------------------------------------------------------------------------


class OCRResult(BaseModel):
    """Result of OCR on a full image."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    engine: str
    language: str


# ---------------------------------------------------------------------------
# VLM Protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# OCR Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ImageOCRBackend(Protocol):
    """Interface for OCR backends (e.g., Tesseract).

    Accepts raw image bytes and returns extracted text with confidence.
    Structurally compatible with ingestkit-forms ``OCRBackend`` so a
    single concrete backend can satisfy both via duck typing.
    """

    def ocr_image(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ) -> OCRResult:
        """Run OCR on the given image bytes. Returns extracted text."""
        ...

    def engine_name(self) -> str:
        """Return the OCR engine identifier (e.g. 'tesseract')."""
        ...


__all__ = [
    "OCRResult",
    "ImageVLMBackend",
    "ImageOCRBackend",
    "EmbeddingBackend",
    "VectorStoreBackend",
]
