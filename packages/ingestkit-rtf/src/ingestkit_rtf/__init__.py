"""ingestkit-rtf -- RTF (Rich Text Format) document converter for RAG pipelines.

Public API re-exports for convenient access.
"""

from ingestkit_rtf.config import RTFProcessorConfig
from ingestkit_rtf.converter import ExtractResult, chunk_text, extract_text
from ingestkit_rtf.errors import ErrorCode, IngestError
from ingestkit_rtf.models import RTFChunkMetadata, ProcessingResult
from ingestkit_rtf.router import RTFRouter
from ingestkit_rtf.security import RTFSecurityScanner

__all__ = [
    "RTFRouter",
    "RTFProcessorConfig",
    "ErrorCode",
    "IngestError",
    "RTFChunkMetadata",
    "ProcessingResult",
    "RTFSecurityScanner",
    "ExtractResult",
    "extract_text",
    "chunk_text",
]
