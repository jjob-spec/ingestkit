"""ingestkit-email -- Email (EML/MSG) document converter for RAG pipelines.

Re-exports all public types: router, config, models, errors, converters,
security scanner, and HTML stripping utility.
"""

from ingestkit_email.config import EmailProcessorConfig
from ingestkit_email.converters.base import EmailContent
from ingestkit_email.converters.eml import EMLConverter
from ingestkit_email.converters.msg import MSGConverter
from ingestkit_email.errors import ErrorCode, IngestError
from ingestkit_email.html_strip import strip_html_tags
from ingestkit_email.models import (
    EmailChunkMetadata,
    EmailContentType,
    EmailMetadata,
    EmailType,
    ProcessingResult,
)
from ingestkit_email.router import EmailRouter, create_default_router
from ingestkit_email.security import EmailSecurityScanner

MIME_TYPES = {".eml": "message/rfc822", ".msg": "application/vnd.ms-outlook"}

__all__ = [
    # Router
    "EmailRouter",
    "create_default_router",
    # Config
    "EmailProcessorConfig",
    # Errors
    "ErrorCode",
    "IngestError",
    # Models
    "EmailType",
    "EmailContentType",
    "EmailMetadata",
    "EmailChunkMetadata",
    "ProcessingResult",
    # Converters
    "EMLConverter",
    "MSGConverter",
    "EmailContent",
    # Security
    "EmailSecurityScanner",
    # Utilities
    "strip_html_tags",
    # Constants
    "MIME_TYPES",
]
