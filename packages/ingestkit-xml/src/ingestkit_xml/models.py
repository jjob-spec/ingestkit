"""Package-specific Pydantic models for the ingestkit-xml package.

Contains ``XMLChunkMetadata``, ``ExtractResult``, and ``ProcessingResult``.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingestkit_core.errors import BaseIngestError
from ingestkit_core.models import BaseChunkMetadata, EmbedStageResult, WrittenArtifacts


class XMLChunkMetadata(BaseChunkMetadata):
    """Chunk metadata with XML-specific fields.

    Extends ``BaseChunkMetadata`` with defaults and fields relevant to
    extracted XML output.
    """

    source_format: str = "xml"
    root_tag: str | None = None
    total_elements: int = 0
    max_depth: int = 0
    namespace_count: int = 0


class ExtractResult(BaseModel):
    """Internal model for the output of ``extract_xml()``."""

    lines: list[str]
    total_elements: int
    max_depth: int
    namespaces: list[str]
    root_tag: str
    truncated: bool
    fallback_used: bool


class ProcessingResult(BaseModel):
    """Final result of XML ingestion via ``XMLRouter.process()``."""

    file_path: str
    ingest_key: str
    ingest_run_id: str
    tenant_id: str | None = None
    embed_result: EmbedStageResult | None = None
    chunks_created: int = 0
    written: WrittenArtifacts = WrittenArtifacts()
    total_elements: int = 0
    max_depth: int = 0
    errors: list[str] = []
    warnings: list[str] = []
    error_details: list[BaseIngestError] = []
    processing_time_seconds: float = 0.0
