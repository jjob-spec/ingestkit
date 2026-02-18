"""Package-specific Pydantic models for the ingestkit-rtf package.

Contains ``RTFChunkMetadata`` and ``ProcessingResult``.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingestkit_core.errors import BaseIngestError
from ingestkit_core.models import BaseChunkMetadata, EmbedStageResult, WrittenArtifacts


class RTFChunkMetadata(BaseChunkMetadata):
    """Chunk metadata with RTF-specific fields.

    Extends ``BaseChunkMetadata`` with defaults and fields relevant to
    RTF document output.
    """

    source_format: str = "rtf"
    word_count: int = 0


class ProcessingResult(BaseModel):
    """Final result of RTF ingestion via ``RTFRouter.process()``."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    embed_result: EmbedStageResult | None = None
    chunks_created: int = 0
    written: WrittenArtifacts = WrittenArtifacts()
    word_count: int = 0
    errors: list[str] = []
    warnings: list[str] = []
    error_details: list[BaseIngestError] = []
    processing_time_seconds: float = 0.0
