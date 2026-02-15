"""OCR engine abstraction layer with Tesseract and PaddleOCR adapters.

Provides a Protocol-based ``OCREngineInterface``, a Pydantic
``OCRPageResult`` model, ``TesseractEngine`` and ``PaddleOCREngine``
adapters, and a ``create_ocr_engine()`` factory with
PaddleOCR-to-Tesseract fallback, per SPEC sections 12.1-12.5.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import OCREngine

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("ingestkit_pdf.utils.ocr_engines")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EngineUnavailableError(Exception):
    """Raised when no OCR engine is available at setup time."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class OCRPageResult(BaseModel):
    """Standardized OCR output from any engine.

    This is the raw engine-level result.  The higher-level ``OCRResult``
    in ``models.py`` wraps this with pipeline context (page number, DPI,
    preprocessing steps).
    """

    text: str
    confidence: float  # 0.0-1.0 average
    word_confidences: list[float] | None = None
    language_detected: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OCREngineInterface(Protocol):
    """Structural subtyping interface for OCR engines."""

    def recognize(self, image: Image.Image, language: str) -> OCRPageResult: ...

    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, str] = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "ru": "rus",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "pl": "pol",
    "tr": "tur",
    "vi": "vie",
    "th": "tha",
    "uk": "ukr",
    "cs": "ces",
    "ro": "ron",
}


def _to_tesseract_lang(language: str) -> str:
    """Map ISO 639-1 language code to Tesseract's ISO 639-3 code."""
    return _LANGUAGE_MAP.get(language, language)


# ---------------------------------------------------------------------------
# PaddleOCR language mapping
# ---------------------------------------------------------------------------

_PADDLEOCR_LANG_MAP: dict[str, str] = {
    "zh": "ch",
    "ko": "korean",
    "ja": "japan",
    "de": "german",
}


def _to_paddleocr_lang(language: str) -> str:
    """Map ISO 639-1 language code to PaddleOCR's expected code.

    PaddleOCR uses ISO 639-1 directly for most languages, with a few
    exceptions (zh -> ch, ko -> korean, ja -> japan, de -> german).
    Unmapped codes pass through as-is.
    """
    return _PADDLEOCR_LANG_MAP.get(language, language)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _tesseract_available() -> bool:
    """Check if Tesseract binary and pytesseract are both available."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tesseract adapter
# ---------------------------------------------------------------------------


class TesseractEngine:
    """Adapter for Tesseract via pytesseract.  Required baseline on all platforms."""

    def __init__(self, lang: str = "eng") -> None:
        self._lang = _to_tesseract_lang(lang)

    def recognize(self, image: Image.Image, language: str) -> OCRPageResult:
        """Run Tesseract OCR on an image.

        Parameters
        ----------
        image:
            PIL Image to recognize.
        language:
            ISO 639-1 language code.  Mapped to Tesseract format internally.
            Overrides the instance default if provided.
        """
        import pytesseract

        tess_lang = _to_tesseract_lang(language) if language else self._lang
        data = pytesseract.image_to_data(
            image, lang=tess_lang, output_type=pytesseract.Output.DICT
        )

        words: list[str] = []
        confidences: list[float] = []
        for text_val, conf_val in zip(data["text"], data["conf"]):
            text_stripped = text_val.strip()
            if text_stripped and conf_val != -1:
                words.append(text_stripped)
                confidences.append(conf_val / 100.0)

        full_text = " ".join(words)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return OCRPageResult(
            text=full_text,
            confidence=avg_confidence,
            word_confidences=confidences if confidences else None,
            language_detected=None,
        )

    def name(self) -> str:
        """Return the engine name."""
        return "tesseract"


# ---------------------------------------------------------------------------
# PaddleOCR adapter
# ---------------------------------------------------------------------------


class PaddleOCREngine:
    """Adapter for PaddleOCR. Optional, higher accuracy (~96.6% vs ~89%).

    PaddleOCR returns line-level text with confidence, unlike Tesseract's
    word-level output.  Each ``ProcessPoolExecutor`` worker must create its
    own instance (known PaddleOCR thread-safety issue per SPEC §12.4).
    """

    def __init__(self, lang: str = "en") -> None:
        import paddleocr as _paddleocr  # noqa: F811

        self._lang = _to_paddleocr_lang(lang)
        self._ocr = _paddleocr.PaddleOCR(
            lang=self._lang,
            use_angle_cls=True,
            show_log=False,
        )

    def recognize(self, image: Image.Image, language: str) -> OCRPageResult:
        """Run PaddleOCR on an image.

        Parameters
        ----------
        image:
            PIL Image to recognize.
        language:
            ISO 639-1 language code.  Mapped to PaddleOCR format internally.
            Overrides the instance default if provided.
        """
        import numpy as np

        paddle_lang = _to_paddleocr_lang(language) if language else self._lang

        # Re-create engine if language differs from init (PaddleOCR is lang-bound)
        if paddle_lang != self._lang:
            import paddleocr as _paddleocr

            ocr = _paddleocr.PaddleOCR(
                lang=paddle_lang,
                use_angle_cls=True,
                show_log=False,
            )
        else:
            ocr = self._ocr

        img_array = np.array(image)
        result = ocr.ocr(img_array, cls=True)

        # Parse nested structure: result = [[line1, line2, ...]]
        # Each line = [bounding_box, (text, confidence)]
        # result can be None or [[None]] for empty pages
        lines: list[str] = []
        confidences: list[float] = []

        if result is not None:
            for page_lines in result:
                if page_lines is None:
                    continue
                for line in page_lines:
                    if line is None:
                        continue
                    # line = [bbox, (text, conf)]
                    text_conf = line[1]
                    text_val = text_conf[0]
                    conf_val = text_conf[1]
                    if text_val.strip():
                        lines.append(text_val.strip())
                        confidences.append(float(conf_val))

        full_text = "\n".join(lines)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return OCRPageResult(
            text=full_text,
            confidence=avg_confidence,
            word_confidences=confidences if confidences else None,
            language_detected=None,
        )

    def name(self) -> str:
        """Return the engine name."""
        return "paddleocr"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ocr_engine(
    config: PDFProcessorConfig,
) -> tuple[OCREngineInterface, list[str]]:
    """Create the configured OCR engine with fallback.

    Returns
    -------
    tuple[OCREngineInterface, list[str]]
        The engine instance and any warning codes emitted during creation.

    Raises
    ------
    EngineUnavailableError
        If Tesseract (the required baseline) is not available.
    """
    warnings: list[str] = []

    if config.ocr_engine == OCREngine.PADDLEOCR:
        try:
            return PaddleOCREngine(lang=config.ocr_language), warnings
        except ImportError:
            warnings.append("W_OCR_ENGINE_FALLBACK")
            logger.warning(
                "PaddleOCR requested but not available, "
                "falling back to Tesseract baseline"
            )

    if not _tesseract_available():
        raise EngineUnavailableError(
            "Tesseract is the required OCR baseline but is not installed. "
            "Install with: apt install tesseract-ocr && pip install pytesseract"
        )

    return TesseractEngine(lang=config.ocr_language), warnings
