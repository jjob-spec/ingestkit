"""Utility modules for ingestkit-pdf."""

from ingestkit_pdf.utils.header_footer import HeaderFooterDetector
from ingestkit_pdf.utils.heading_detector import HeadingDetector
from ingestkit_pdf.utils.language import detect_language, map_language_to_ocr
from ingestkit_pdf.utils.ocr_engines import (
    EngineUnavailableError,
    OCREngineInterface,
    OCRPageResult,
    TesseractEngine,
    create_ocr_engine,
)
from ingestkit_pdf.utils.ocr_postprocess import postprocess_ocr_text
from ingestkit_pdf.utils.page_renderer import PageRenderer

__all__ = [
    "EngineUnavailableError",
    "HeaderFooterDetector",
    "HeadingDetector",
    "OCREngineInterface",
    "OCRPageResult",
    "PageRenderer",
    "TesseractEngine",
    "create_ocr_engine",
    "detect_language",
    "map_language_to_ocr",
    "postprocess_ocr_text",
]
