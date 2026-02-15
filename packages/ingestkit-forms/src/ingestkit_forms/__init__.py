"""ingestkit-forms -- Template-driven form extraction plugin for ingestkit.

Public API exports for form template matching, extraction, and output.
Stub module -- implementations will be added in subsequent issues.
"""

from ingestkit_forms.config import FormProcessorConfig, RedactTarget
from ingestkit_forms.errors import FormErrorCode, FormIngestError
from ingestkit_forms.protocols import (
    EmbeddingBackend,
    FormTemplateStore,
    OCRBackend,
    OCRRegionResult,
    PDFWidgetBackend,
    StructuredDBBackend,
    VectorStoreBackend,
    VLMBackend,
    VLMFieldResult,
    WidgetField,
)
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
]
