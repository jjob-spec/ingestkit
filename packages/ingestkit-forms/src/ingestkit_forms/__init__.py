"""ingestkit-forms -- Template-driven form extraction plugin for ingestkit.

Public API exports for form template matching, extraction, and output.
"""

from ingestkit_forms.api import FormTemplateAPI
from ingestkit_forms.config import FormProcessorConfig, RedactTarget
from ingestkit_forms.errors import FormErrorCode, FormIngestError, FormIngestException
from ingestkit_forms.matcher import (
    FormMatcher,
    compute_layout_fingerprint,
    compute_layout_fingerprint_from_file,
    compute_layout_similarity,
    detect_source_format,
)
from ingestkit_forms.protocols import (
    EmbeddingBackend,
    FormTemplateStore,
    LayoutFingerprinter,
    OCRBackend,
    OCRRegionResult,
    PDFWidgetBackend,
    StructuredDBBackend,
    VectorStoreBackend,
    VLMBackend,
    VLMFieldResult,
    WidgetField,
)
from ingestkit_forms.stores import FileSystemTemplateStore
from ingestkit_forms.models import (
    BoundingBox,
    CellAddress,
    DualWriteMode,
    ExtractionPreview,
    ExtractedField,
    FieldMapping,
    FieldType,
    FormChunkMetadata,
    FormChunkPayload,
    FormExtractionResult,
    FormIngestRequest,
    FormProcessingResult,
    FormTemplate,
    FormTemplateCreateRequest,
    FormTemplateUpdateRequest,
    FormWrittenArtifacts,
    RollbackResult,
    SourceFormat,
    TemplateMatch,
)

__all__: list[str] = [
    # Errors and config (from issues #56, #58)
    "FormErrorCode",
    "FormIngestError",
    "FormIngestException",
    "FormProcessorConfig",
    "RedactTarget",
    # Enums
    "SourceFormat",
    "FieldType",
    "DualWriteMode",
    # Template models
    "BoundingBox",
    "CellAddress",
    "FieldMapping",
    "FormTemplate",
    # Matching models
    "TemplateMatch",
    "FormIngestRequest",
    # Extraction models
    "ExtractedField",
    "FormExtractionResult",
    # Chunk models
    "FormChunkMetadata",
    "FormChunkPayload",
    # Result models
    "FormWrittenArtifacts",
    "FormProcessingResult",
    "RollbackResult",
    # Request/Response models
    "FormTemplateCreateRequest",
    "FormTemplateUpdateRequest",
    "ExtractionPreview",
    # Protocols (from issue #60)
    "FormTemplateStore",
    "OCRBackend",
    "PDFWidgetBackend",
    "VLMBackend",
    # Protocol result models
    "OCRRegionResult",
    "WidgetField",
    "VLMFieldResult",
    # Re-exported from ingestkit-core
    "VectorStoreBackend",
    "StructuredDBBackend",
    "EmbeddingBackend",
    # API (from issue #61)
    "FormTemplateAPI",
    # Matcher (from issue #61)
    "compute_layout_fingerprint",
    "compute_layout_similarity",
    "compute_layout_fingerprint_from_file",
    # Matcher (from issue #62)
    "FormMatcher",
    "detect_source_format",
    # Protocol additions (from issue #62)
    "LayoutFingerprinter",
    # Stores (from issue #61)
    "FileSystemTemplateStore",
]
