"""ingestkit-excel -- Excel file ingestion plugin for the ingestkit framework.

Public API exports for models, enums, errors, configuration, and backend
protocols.  Higher-level components (``ExcelRouter``, ``create_default_router``)
will be exported here once implemented in subsequent issues.
"""

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.idempotency import compute_ingest_key
from ingestkit_excel.models import (
    ChunkMetadata,
    ChunkPayload,
    ClassificationResult,
    ClassificationStageResult,
    ClassificationTier,
    EmbedStageResult,
    FileProfile,
    FileType,
    IngestKey,
    IngestionMethod,
    ParseStageResult,
    ParserUsed,
    ProcessingResult,
    RegionType,
    SheetProfile,
    SheetRegion,
    WrittenArtifacts,
)
from ingestkit_excel.inspector import ExcelInspector
from ingestkit_excel.llm_classifier import LLMClassifier
from ingestkit_excel.parser_chain import ParserChain
from ingestkit_excel.processors import HybridSplitter, StructuredDBProcessor, TextSerializer
from ingestkit_excel.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

__all__ = [
    # Enums
    "FileType",
    "ClassificationTier",
    "IngestionMethod",
    "RegionType",
    "ParserUsed",
    # Idempotency
    "IngestKey",
    "compute_ingest_key",
    # Stage artifacts
    "ParseStageResult",
    "ClassificationStageResult",
    "EmbedStageResult",
    # Core models
    "SheetProfile",
    "FileProfile",
    "ClassificationResult",
    "ChunkMetadata",
    "ChunkPayload",
    "SheetRegion",
    "WrittenArtifacts",
    "ProcessingResult",
    # Parser
    "ParserChain",
    # Inspector
    "ExcelInspector",
    # LLM Classifier
    "LLMClassifier",
    # Processors
    "StructuredDBProcessor",
    "TextSerializer",
    "HybridSplitter",
    # Errors
    "ErrorCode",
    "IngestError",
    # Config
    "ExcelProcessorConfig",
    # Protocols
    "VectorStoreBackend",
    "StructuredDBBackend",
    "LLMBackend",
    "EmbeddingBackend",
]
