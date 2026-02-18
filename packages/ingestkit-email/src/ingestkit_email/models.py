"""Pydantic models and enumerations for the ingestkit-email package.

Contains email-specific types: ``EmailType``, ``EmailContentType``,
``EmailMetadata``, ``EmailChunkMetadata``, and ``ProcessingResult``.
Re-exports shared types from ``ingestkit_core.models``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ingestkit_core.models import (
    BaseChunkMetadata,
    ChunkPayload,
    EmbedStageResult,
    IngestKey,
    WrittenArtifacts,
)

from ingestkit_email.errors import IngestError

# Re-exports for convenience
__all__ = [
    "IngestKey",
    "BaseChunkMetadata",
    "ChunkPayload",
    "WrittenArtifacts",
    "EmbedStageResult",
    "EmailType",
    "EmailContentType",
    "EmailMetadata",
    "EmailChunkMetadata",
    "ProcessingResult",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EmailType(str, Enum):
    """Source format of the email file."""

    EML = "eml"
    MSG = "msg"


class EmailContentType(str, Enum):
    """How the body text was obtained."""

    PLAIN_TEXT = "plain_text"
    HTML_CONVERTED = "html_converted"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class EmailMetadata(BaseModel):
    """Extracted metadata from an email file."""

    from_address: str | None = None
    to_address: str | None = None
    cc_address: str | None = None
    date: str | None = None
    subject: str | None = None
    message_id: str | None = None
    content_type: EmailContentType = EmailContentType.PLAIN_TEXT
    attachment_count: int = 0
    has_html_body: bool = False
    has_plain_body: bool = False


# ---------------------------------------------------------------------------
# Chunk Metadata
# ---------------------------------------------------------------------------


class EmailChunkMetadata(BaseChunkMetadata):
    """Chunk metadata extended with email-specific fields."""

    source_format: str = "email"
    email_type: EmailType | None = None
    from_address: str | None = None
    to_address: str | None = None
    subject: str | None = None
    date: str | None = None
    message_id: str | None = None
    content_type: EmailContentType = EmailContentType.PLAIN_TEXT


# ---------------------------------------------------------------------------
# Processing Result
# ---------------------------------------------------------------------------


class ProcessingResult(BaseModel):
    """Final result of processing an email file through the pipeline."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    email_metadata: EmailMetadata | None = None
    embed_result: EmbedStageResult | None = None
    chunks_created: int = 0
    written: WrittenArtifacts = WrittenArtifacts()
    errors: list[str] = []
    warnings: list[str] = []
    error_details: list[IngestError] = []
    processing_time_seconds: float = 0.0
