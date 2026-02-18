"""ingestkit-image -- VLM captioning and OCR text extraction for RAG indexing."""

from ingestkit_image.caption import CaptionError, ImageCaptionConverter
from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.errors import ImageErrorCode, ImageIngestError
from ingestkit_image.models import (
    CaptionResult,
    ImageChunkMetadata,
    ImageMetadata,
    ImageProcessingResult,
    ImageType,
    OCRTextResult,
)
from ingestkit_image.ocr_extract import ImageOCRExtractor, OCRExtractError
from ingestkit_image.protocols import (
    EmbeddingBackend,
    ImageOCRBackend,
    ImageVLMBackend,
    OCRResult,
    VectorStoreBackend,
)
from ingestkit_image.router import ImageRouter
from ingestkit_image.security import ImageSecurityScanner

__all__ = [
    # Router
    "ImageRouter",
    # Config
    "ImageProcessorConfig",
    # Converters
    "ImageCaptionConverter",
    "CaptionError",
    "ImageOCRExtractor",
    "OCRExtractError",
    # Security
    "ImageSecurityScanner",
    # Models -- enums
    "ImageType",
    "ImageErrorCode",
    # Models -- data
    "ImageMetadata",
    "ImageChunkMetadata",
    "CaptionResult",
    "OCRTextResult",
    "ImageProcessingResult",
    # Errors
    "ImageIngestError",
    # Protocols
    "ImageVLMBackend",
    "ImageOCRBackend",
    "OCRResult",
    "EmbeddingBackend",
    "VectorStoreBackend",
]
