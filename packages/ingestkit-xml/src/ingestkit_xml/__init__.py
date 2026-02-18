"""ingestkit-xml -- XML document converter for RAG pipelines.

Public API re-exports for convenient access.
"""

from ingestkit_xml.config import XMLProcessorConfig
from ingestkit_xml.converter import chunk_text, extract_xml
from ingestkit_xml.errors import ErrorCode, IngestError
from ingestkit_xml.models import ExtractResult, ProcessingResult, XMLChunkMetadata
from ingestkit_xml.router import XMLRouter
from ingestkit_xml.security import XMLSecurityScanner

__all__ = [
    "XMLRouter",
    "XMLProcessorConfig",
    "ErrorCode",
    "IngestError",
    "XMLChunkMetadata",
    "ProcessingResult",
    "ExtractResult",
    "XMLSecurityScanner",
    "extract_xml",
    "chunk_text",
]
