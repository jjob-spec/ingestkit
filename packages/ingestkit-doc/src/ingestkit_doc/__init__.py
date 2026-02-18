"""ingestkit-doc -- Legacy Word (.doc) document converter for RAG pipelines.

Public API re-exports for convenient access.
"""

from ingestkit_doc.config import DocProcessorConfig
from ingestkit_doc.converter import ExtractResult, chunk_text, extract_text
from ingestkit_doc.errors import ErrorCode, IngestError
from ingestkit_doc.models import DocChunkMetadata, ProcessingResult
from ingestkit_doc.router import DocRouter
from ingestkit_doc.security import DocSecurityScanner

__all__ = [
    "DocRouter",
    "DocProcessorConfig",
    "ErrorCode",
    "IngestError",
    "DocChunkMetadata",
    "ProcessingResult",
    "DocSecurityScanner",
    "ExtractResult",
    "extract_text",
    "chunk_text",
]
