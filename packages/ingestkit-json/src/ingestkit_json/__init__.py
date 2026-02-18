"""ingestkit-json -- JSON document converter for RAG pipelines.

Public API re-exports for convenient access.
"""

from ingestkit_json.config import JSONProcessorConfig
from ingestkit_json.converter import chunk_text, flatten_json
from ingestkit_json.errors import ErrorCode, IngestError
from ingestkit_json.models import FlattenResult, JSONChunkMetadata, ProcessingResult
from ingestkit_json.router import JSONRouter
from ingestkit_json.security import JSONSecurityScanner

__all__ = [
    "JSONRouter",
    "JSONProcessorConfig",
    "ErrorCode",
    "IngestError",
    "JSONChunkMetadata",
    "ProcessingResult",
    "FlattenResult",
    "JSONSecurityScanner",
    "flatten_json",
    "chunk_text",
]
