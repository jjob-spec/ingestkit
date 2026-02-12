"""ingestkit-core -- Shared primitives for the ingestkit framework.

Re-exports all public types: errors, models, protocols, and utilities.
"""

from ingestkit_core.errors import BaseIngestError, CoreErrorCode
from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import (
    BaseChunkMetadata,
    ChunkPayload,
    ClassificationTier,
    EmbedStageResult,
    IngestKey,
    WrittenArtifacts,
)
from ingestkit_core.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

__all__ = [
    # Errors
    "CoreErrorCode",
    "BaseIngestError",
    # Models
    "ClassificationTier",
    "IngestKey",
    "EmbedStageResult",
    "BaseChunkMetadata",
    "ChunkPayload",
    "WrittenArtifacts",
    # Idempotency
    "compute_ingest_key",
    # Protocols
    "VectorStoreBackend",
    "StructuredDBBackend",
    "LLMBackend",
    "EmbeddingBackend",
]
