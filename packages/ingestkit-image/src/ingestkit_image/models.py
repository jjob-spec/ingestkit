"""Pydantic models and enumerations for the ingestkit-image package.

Contains image-specific types: ``ImageType``, ``ImageMetadata``,
``ImageChunkMetadata``, ``CaptionResult``, and ``ImageProcessingResult``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ingestkit_core.models import (
    BaseChunkMetadata,
    ChunkPayload,
    EmbedStageResult,
    WrittenArtifacts,
)

from ingestkit_image.errors import ImageIngestError


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ImageType(str, Enum):
    """Detected image format."""

    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class ImageMetadata(BaseModel):
    """Metadata extracted from the image file."""

    file_path: str
    file_size_bytes: int
    image_type: ImageType
    width: int
    height: int
    content_hash: str
    has_exif: bool = False
    color_mode: str | None = None  # RGB, RGBA, L, etc.


class ImageChunkMetadata(BaseChunkMetadata):
    """Chunk metadata for image captions. Extends core BaseChunkMetadata."""

    source_format: str = "image"
    source_type: str = "image_caption"
    image_type: str | None = None  # jpeg, png, etc.
    image_width: int | None = None
    image_height: int | None = None
    vlm_model: str | None = None
    caption_prompt: str | None = None


# ---------------------------------------------------------------------------
# Stage Results
# ---------------------------------------------------------------------------


class CaptionResult(BaseModel):
    """Output of the VLM captioning stage."""

    caption: str
    model_used: str
    caption_duration_seconds: float


# ---------------------------------------------------------------------------
# Processing Result
# ---------------------------------------------------------------------------


class ImageProcessingResult(BaseModel):
    """Final result returned after processing an image."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    image_metadata: ImageMetadata | None = None
    caption_result: CaptionResult | None = None
    embed_result: EmbedStageResult | None = None
    chunks_created: int
    written: WrittenArtifacts
    errors: list[str]
    warnings: list[str]
    error_details: list[ImageIngestError] = []
    processing_time_seconds: float


# Re-export shared types for convenience
__all__ = [
    "ImageType",
    "ImageMetadata",
    "ImageChunkMetadata",
    "CaptionResult",
    "ImageProcessingResult",
    "BaseChunkMetadata",
    "ChunkPayload",
    "EmbedStageResult",
    "WrittenArtifacts",
]
