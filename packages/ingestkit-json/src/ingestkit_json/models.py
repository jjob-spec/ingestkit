"""Package-specific Pydantic models for the ingestkit-json package.

Contains ``JSONChunkMetadata``, ``FlattenResult``, and ``ProcessingResult``.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingestkit_core.errors import BaseIngestError
from ingestkit_core.models import BaseChunkMetadata, EmbedStageResult, WrittenArtifacts


class JSONChunkMetadata(BaseChunkMetadata):
    """Chunk metadata with JSON-specific fields.

    Extends ``BaseChunkMetadata`` with defaults and fields relevant to
    flattened JSON output.
    """

    source_format: str = "json"
    key_path_prefix: str | None = None
    total_keys: int = 0
    nesting_depth: int = 0


class FlattenResult(BaseModel):
    """Internal model for the output of ``flatten_json()``."""

    lines: list[str]
    total_keys: int
    max_depth: int
    truncated: bool


class ProcessingResult(BaseModel):
    """Final result of JSON ingestion via ``JSONRouter.process()``."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    embed_result: EmbedStageResult | None = None
    chunks_created: int = 0
    written: WrittenArtifacts = WrittenArtifacts()
    total_keys: int = 0
    max_depth: int = 0
    errors: list[str] = []
    warnings: list[str] = []
    error_details: list[BaseIngestError] = []
    processing_time_seconds: float = 0.0
