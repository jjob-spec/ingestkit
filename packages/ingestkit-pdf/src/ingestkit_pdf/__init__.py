"""ingestkit-pdf -- Tiered PDF file processing for RAG pipelines."""

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.llm_classifier import LLMClassificationResponse, PDFLLMClassifier
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationStageResult,
    ClassificationTier,
    ChunkPayload,
    DocumentMetadata,
    DocumentProfile,
    EmbedStageResult,
    ExtractionQuality,
    IngestionMethod,
    OCRResult,
    OCRStageResult,
    PageProfile,
    PageType,
    PDFChunkMetadata,
    PDFType,
    ParseStageResult,
    ProcessingResult,
    TableResult,
    WrittenArtifacts,
)
from ingestkit_pdf.processors.ocr_processor import OCRProcessor
from ingestkit_pdf.processors.text_extractor import TextExtractor
from ingestkit_pdf.execution import (
    DistributedExecutionBackend,
    ExecutionBackend,
    ExecutionError,
    LocalExecutionBackend,
)
from ingestkit_pdf.router import PDFRouter, create_default_router

__all__ = [
    # Router
    "PDFRouter",
    "create_default_router",
    # Config
    "PDFProcessorConfig",
    # Models -- enums
    "PDFType",
    "PageType",
    "ClassificationTier",
    "IngestionMethod",
    # Models -- data
    "ClassificationResult",
    "ProcessingResult",
    "ChunkPayload",
    "PDFChunkMetadata",
    "DocumentProfile",
    "DocumentMetadata",
    "PageProfile",
    "ExtractionQuality",
    "OCRResult",
    "TableResult",
    "WrittenArtifacts",
    # Stage results
    "ParseStageResult",
    "ClassificationStageResult",
    "OCRStageResult",
    "EmbedStageResult",
    # Errors
    "ErrorCode",
    "IngestError",
    # Classifiers
    "PDFInspector",
    "PDFLLMClassifier",
    "LLMClassificationResponse",
    # Processors
    "TextExtractor",
    "OCRProcessor",
    # Execution backends
    "ExecutionBackend",
    "ExecutionError",
    "LocalExecutionBackend",
    "DistributedExecutionBackend",
]
