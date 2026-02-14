"""Tests for ingestkit_pdf.utils.language — language detection utilities."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_pdf.models import OCREngine
from ingestkit_pdf.utils.language import (
    DEFAULT_PADDLEOCR_LANG,
    DEFAULT_TESSERACT_LANG,
    detect_language,
    map_language_to_ocr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_mock_langdetect(return_value: list[dict] | None = None,
                              side_effect: Exception | None = None) -> MagicMock:
    """Install a mock ``fast_langdetect`` module in ``sys.modules``.

    Returns the mock ``detect`` function so callers can assert on it.
    """
    mock_detect = MagicMock()
    if side_effect is not None:
        mock_detect.side_effect = side_effect
    elif return_value is not None:
        mock_detect.return_value = return_value

    mock_module = ModuleType("fast_langdetect")
    mock_module.detect = mock_detect  # type: ignore[attr-defined]
    sys.modules["fast_langdetect"] = mock_module
    return mock_detect


def _remove_mock_langdetect() -> None:
    """Remove the mock ``fast_langdetect`` module from ``sys.modules``."""
    sys.modules.pop("fast_langdetect", None)


# ---------------------------------------------------------------------------
# detect_language tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectLanguage:
    """Tests for the detect_language function."""

    def test_empty_string(self) -> None:
        lang, score = detect_language("")
        assert lang == "en"
        assert score == 1.0

    def test_whitespace_only(self) -> None:
        lang, score = detect_language("   \n\t  ")
        assert lang == "en"
        assert score == 1.0

    def test_very_short_text(self) -> None:
        lang, score = detect_language("Hi")
        assert lang == "en"
        assert score == 0.5

    def test_custom_default_language_empty(self) -> None:
        lang, score = detect_language("", default_language="fr")
        assert lang == "fr"
        assert score == 1.0

    def test_custom_default_language_short(self) -> None:
        lang, score = detect_language("Bonjour", default_language="fr")
        assert lang == "fr"
        assert score == 0.5

    def test_english_detection(self) -> None:
        mock_detect = _install_mock_langdetect(
            return_value=[{"lang": "en", "score": 0.99}]
        )
        try:
            lang, score = detect_language(
                "This is a sample English paragraph with enough text"
            )
            assert lang == "en"
            assert score == 0.99
            mock_detect.assert_called_once()
        finally:
            _remove_mock_langdetect()

    def test_spanish_detection(self) -> None:
        mock_detect = _install_mock_langdetect(
            return_value=[{"lang": "es", "score": 0.95}]
        )
        try:
            lang, score = detect_language(
                "Este es un parrafo de ejemplo en espanol para detectar"
            )
            assert lang == "es"
            assert score == 0.95
        finally:
            _remove_mock_langdetect()

    def test_import_error_graceful_degradation(self) -> None:
        # Ensure fast_langdetect is NOT in sys.modules
        saved = sys.modules.pop("fast_langdetect", None)
        try:
            # Patch builtins.__import__ to raise ImportError
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
                if name == "fast_langdetect":
                    raise ImportError("No module named 'fast_langdetect'")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = fake_import  # type: ignore[assignment]
            lang, score = detect_language(
                "This is long enough text for detection to proceed"
            )
            assert lang == "en"
            assert score == 0.0
        finally:
            builtins.__import__ = real_import  # type: ignore[assignment]
            if saved is not None:
                sys.modules["fast_langdetect"] = saved

    def test_unexpected_error(self) -> None:
        _install_mock_langdetect(
            side_effect=RuntimeError("something went wrong")
        )
        try:
            lang, score = detect_language(
                "This text is long enough to attempt detection"
            )
            assert lang == "en"
            assert score == 0.0
        finally:
            _remove_mock_langdetect()

    def test_zh_cn_normalization(self) -> None:
        _install_mock_langdetect(
            return_value=[{"lang": "zh-cn", "score": 0.95}]
        )
        try:
            lang, score = detect_language(
                "This text is long enough to trigger detection"
            )
            assert lang == "zh"
            assert score == 0.95
        finally:
            _remove_mock_langdetect()

    def test_zh_tw_normalization(self) -> None:
        _install_mock_langdetect(
            return_value=[{"lang": "zh-tw", "score": 0.92}]
        )
        try:
            lang, score = detect_language(
                "This text is long enough to trigger detection"
            )
            assert lang == "zh"
            assert score == 0.92
        finally:
            _remove_mock_langdetect()

    def test_uses_lite_model(self) -> None:
        mock_detect = _install_mock_langdetect(
            return_value=[{"lang": "en", "score": 0.99}]
        )
        try:
            detect_language("Sufficiently long text for language detection")
            mock_detect.assert_called_once_with(
                "Sufficiently long text for language detection",
                model="lite",
                k=1,
            )
        finally:
            _remove_mock_langdetect()


# ---------------------------------------------------------------------------
# map_language_to_ocr tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMapLanguageToOcr:
    """Tests for the map_language_to_ocr function."""

    def test_english_to_tesseract(self) -> None:
        assert map_language_to_ocr("en", OCREngine.TESSERACT) == "eng"

    def test_english_to_paddleocr(self) -> None:
        assert map_language_to_ocr("en", OCREngine.PADDLEOCR) == "en"

    def test_chinese_to_tesseract(self) -> None:
        assert map_language_to_ocr("zh", OCREngine.TESSERACT) == "chi_sim"

    def test_chinese_to_paddleocr(self) -> None:
        assert map_language_to_ocr("zh", OCREngine.PADDLEOCR) == "ch"

    def test_spanish_to_tesseract(self) -> None:
        assert map_language_to_ocr("es", OCREngine.TESSERACT) == "spa"

    def test_unknown_lang_tesseract_fallback(self) -> None:
        result = map_language_to_ocr("xx", OCREngine.TESSERACT)
        assert result == DEFAULT_TESSERACT_LANG

    def test_unknown_lang_paddleocr_passthrough(self) -> None:
        result = map_language_to_ocr("xx", OCREngine.PADDLEOCR)
        assert result == "xx"

    def test_japanese_to_tesseract(self) -> None:
        assert map_language_to_ocr("ja", OCREngine.TESSERACT) == "jpn"

    def test_japanese_to_paddleocr(self) -> None:
        assert map_language_to_ocr("ja", OCREngine.PADDLEOCR) == "japan"
