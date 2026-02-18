"""ingestkit-image -- VLM-based image captioning for RAG indexing."""

from ingestkit_image.caption import CaptionError, ImageCaptionConverter
from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import (
    CaptionResult,
    ImageChunkMetadata,
    ImageMetadata,
    ImageProcessingResult,
    ImageType,
)
from ingestkit_image.protocols import (
    EmbeddingBackend,
    ImageVLMBackend,
    VectorStoreBackend,
)
from ingestkit_image.router import ImageRouter
from ingestkit_image.security import ImageSecurityScanner

__all__ = [
    # Router
    "ImageRouter",
    # Config
    "ImageProcessorConfig",
    # Converter
    "ImageCaptionConverter",
    "CaptionError",
    # Security
    "ImageSecurityScanner",
    # Models -- enums
    "ImageType",
    "ImageErrorCode",
    # Models -- data
    "ImageMetadata",
    "ImageChunkMetadata",
    "CaptionResult",
    "ImageProcessingResult",
    # Errors
    "ImageIngestError",
    # Protocols
    "ImageVLMBackend",
    "EmbeddingBackend",
    "VectorStoreBackend",
]
