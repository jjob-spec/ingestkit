"""ingestkit-xls -- Legacy Excel (.xls) document converter for RAG pipelines.

Public API re-exports for convenient access.
"""

from ingestkit_xls.config import XlsProcessorConfig
from ingestkit_xls.converter import ExtractResult, chunk_text, extract_sheets
from ingestkit_xls.errors import ErrorCode, IngestError
from ingestkit_xls.models import ProcessingResult, XlsChunkMetadata
from ingestkit_xls.router import XlsRouter
from ingestkit_xls.security import XlsSecurityScanner

__all__ = [
    "XlsRouter",
    "XlsProcessorConfig",
    "ErrorCode",
    "IngestError",
    "XlsChunkMetadata",
    "ProcessingResult",
    "XlsSecurityScanner",
    "ExtractResult",
    "extract_sheets",
    "chunk_text",
]
