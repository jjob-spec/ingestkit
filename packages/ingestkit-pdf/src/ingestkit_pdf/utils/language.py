"""Language detection utilities for OCR routing (SPEC §16.1-§16.3).

Detects page language to route to the correct OCR language model.
Uses ``fast-langdetect`` (FastText wrapper) when available, with graceful
degradation to a configurable default language when the optional dependency
is not installed.

Public API
----------
- ``detect_language(text, *, default_language="en") -> tuple[str, float]``
- ``map_language_to_ocr(lang, engine) -> str``
"""

from __future__ import annotations

import logging

from ingestkit_pdf.models import OCREngine

logger = logging.getLogger("ingestkit_pdf.utils.language")

# ---------------------------------------------------------------------------
# Language mapping constants
# ---------------------------------------------------------------------------

# ISO 639-1 -> Tesseract ISO 639-3 mapping (common languages for HR/IT docs)
_TESSERACT_LANG_MAP: dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "pt": "por",
    "it": "ita",
    "nl": "nld",
    "ru": "rus",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "vi": "vie",
    "th": "tha",
    "pl": "pol",
    "tr": "tur",
    "uk": "ukr",
    "sv": "swe",
    "da": "dan",
    "no": "nor",
    "fi": "fin",
    "cs": "ces",
    "ro": "ron",
    "hu": "hun",
    "el": "ell",
    "he": "heb",
    "id": "ind",
    "ms": "msa",
    "tl": "tgl",
}

# PaddleOCR uses ISO 639-1 directly for most languages, with a few exceptions
_PADDLEOCR_LANG_MAP: dict[str, str] = {
    "zh": "ch",
    "ko": "korean",
    "ja": "japan",
    "ar": "ar",
    "hi": "hi",
    "en": "en",
    "fr": "fr",
    "de": "german",
    "es": "es",
    "pt": "pt",
    "it": "it",
    "ru": "ru",
}

DEFAULT_TESSERACT_LANG = "eng"
DEFAULT_PADDLEOCR_LANG = "en"

# Minimum text length for reliable language detection
_MIN_TEXT_LENGTH = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_language(
    text: str,
    *,
    default_language: str = "en",
) -> tuple[str, float]:
    """Detect the language of *text* using ``fast-langdetect``.

    Parameters
    ----------
    text:
        Input text to classify.
    default_language:
        ISO 639-1 code returned when detection is impossible (empty text,
        missing dependency, etc.).

    Returns
    -------
    tuple[str, float]
        ``(iso_639_1_code, confidence)`` where confidence is in ``[0, 1]``.
    """
    if not text or not text.strip():
        return (default_language, 1.0)

    stripped = text.strip()
    if len(stripped) < _MIN_TEXT_LENGTH:
        logger.warning(
            "Text too short for reliable language detection (%d chars), "
            "returning default '%s'",
            len(stripped),
            default_language,
        )
        return (default_language, 0.5)

    try:
        from fast_langdetect import detect  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "fast-langdetect not installed, returning default language '%s'",
            default_language,
        )
        return (default_language, 0.0)

    try:
        result = detect(stripped, model="lite", k=1)
        lang: str = result[0]["lang"]
        score: float = result[0]["score"]

        # Normalize BCP-47 tags (e.g. zh-cn -> zh) to ISO 639-1
        if "-" in lang:
            lang = lang.split("-")[0]

        return (lang, score)
    except Exception:
        logger.exception(
            "Unexpected error during language detection, "
            "returning default '%s'",
            default_language,
        )
        return (default_language, 0.0)


def map_language_to_ocr(lang: str, engine: OCREngine) -> str:
    """Map an ISO 639-1 language code to the engine-specific OCR code.

    Parameters
    ----------
    lang:
        ISO 639-1 language code (e.g. ``"en"``, ``"zh"``).
    engine:
        Target OCR engine.

    Returns
    -------
    str
        Engine-specific language code.
    """
    if engine == OCREngine.TESSERACT:
        mapped = _TESSERACT_LANG_MAP.get(lang)
        if mapped is not None:
            return mapped
        logger.warning(
            "Unmapped language '%s' for Tesseract, falling back to '%s'",
            lang,
            DEFAULT_TESSERACT_LANG,
        )
        return DEFAULT_TESSERACT_LANG

    # PaddleOCR
    mapped = _PADDLEOCR_LANG_MAP.get(lang)
    if mapped is not None:
        return mapped
    logger.debug(
        "No explicit PaddleOCR mapping for '%s', passing through as-is",
        lang,
    )
    return lang
