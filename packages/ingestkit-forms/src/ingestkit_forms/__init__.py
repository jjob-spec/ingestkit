"""ingestkit-forms -- Template-driven form extraction plugin for ingestkit.

Public API exports for form template matching, extraction, and output.
"""

from ingestkit_forms.api import FormTemplateAPI
from ingestkit_forms.confidence import (
    apply_confidence_actions,
    compute_field_confidence,
    compute_overall_confidence,
)
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
    FormDBBackend,
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
from ingestkit_forms.idempotency import (
    compute_form_extraction_key,
    compute_ingest_key,
    compute_vector_point_id,
)
from ingestkit_forms.router import FormRouter, create_default_router
from ingestkit_forms.security import (
    FormSecurityScanner,
    regex_match_with_timeout,
    validate_table_name,
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
    TemplateStatus,
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
    "TemplateStatus",
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
    "FormDBBackend",
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
    # Confidence scoring (from issue #67)
    "compute_field_confidence",
    "compute_overall_confidence",
    "apply_confidence_actions",
    # Idempotency keying (from issue #71)
    "compute_ingest_key",
    "compute_form_extraction_key",
    "compute_vector_point_id",
    # Security (from issue #70)
    "FormSecurityScanner",
    "regex_match_with_timeout",
    "validate_table_name",
    # Router (from issue #69)
    "FormRouter",
    "create_default_router",
]
