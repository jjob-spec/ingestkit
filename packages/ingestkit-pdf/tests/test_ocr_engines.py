"""Tests for ingestkit_pdf.utils.ocr_engines — OCR engine abstraction layer."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.models import OCREngine
from ingestkit_pdf.utils.ocr_engines import (
    EngineUnavailableError,
    OCREngineInterface,
    OCRPageResult,
    PaddleOCREngine,
    TesseractEngine,
    _LANGUAGE_MAP,
    _PADDLEOCR_LANG_MAP,
    _tesseract_available,
    _to_paddleocr_lang,
    _to_tesseract_lang,
    create_ocr_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_tesseract_data(
    texts: list[str],
    confs: list[int],
) -> dict[str, list]:
    """Build a mock pytesseract.image_to_data return dict."""
    return {"text": texts, "conf": confs}


def _make_config(**overrides: Any) -> PDFProcessorConfig:
    """Create a config with optional overrides."""
    return PDFProcessorConfig(**overrides)


# ---------------------------------------------------------------------------
# TestOCRPageResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCRPageResult:
    """Tests for OCRPageResult Pydantic model."""

    def test_valid_result_creation(self) -> None:
        result = OCRPageResult(
            text="Hello world",
            confidence=0.95,
            word_confidences=[0.9, 1.0],
            language_detected="en",
        )
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.word_confidences == [0.9, 1.0]
        assert result.language_detected == "en"

    def test_minimal_result(self) -> None:
        result = OCRPageResult(text="Hello", confidence=0.8)
        assert result.text == "Hello"
        assert result.confidence == 0.8
        assert result.word_confidences is None
        assert result.language_detected is None

    def test_empty_text_is_valid(self) -> None:
        result = OCRPageResult(text="", confidence=0.0)
        assert result.text == ""

    def test_word_confidences_list(self) -> None:
        confs = [0.9, 0.8, 0.95]
        result = OCRPageResult(text="a b c", confidence=0.88, word_confidences=confs)
        assert result.word_confidences == [0.9, 0.8, 0.95]

    def test_language_detected(self) -> None:
        result = OCRPageResult(text="Bonjour", confidence=0.9, language_detected="fr")
        assert result.language_detected == "fr"

    def test_confidence_not_bounded(self) -> None:
        """No ge/le validators — raw float accepted; consumer validates range."""
        result = OCRPageResult(text="test", confidence=1.5)
        assert result.confidence == 1.5
        result2 = OCRPageResult(text="test", confidence=-0.1)
        assert result2.confidence == -0.1


# ---------------------------------------------------------------------------
# TestOCREngineInterface
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCREngineInterface:
    """Tests for the OCREngineInterface Protocol."""

    def test_tesseract_engine_satisfies_protocol(self) -> None:
        engine = TesseractEngine()
        assert isinstance(engine, OCREngineInterface)

    def test_paddleocr_engine_satisfies_protocol(self) -> None:
        mock_module = MagicMock()
        mock_module.PaddleOCR.return_value = MagicMock()
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            engine = PaddleOCREngine()
            assert isinstance(engine, OCREngineInterface)

    def test_arbitrary_class_without_methods_fails_protocol(self) -> None:
        assert not isinstance(object(), OCREngineInterface)


# ---------------------------------------------------------------------------
# TestLanguageMapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLanguageMapping:
    """Tests for _to_tesseract_lang() and _LANGUAGE_MAP."""

    def test_english_maps_to_eng(self) -> None:
        assert _to_tesseract_lang("en") == "eng"

    def test_french_maps_to_fra(self) -> None:
        assert _to_tesseract_lang("fr") == "fra"

    def test_chinese_maps_to_chi_sim(self) -> None:
        assert _to_tesseract_lang("zh") == "chi_sim"

    def test_unknown_code_passes_through(self) -> None:
        assert _to_tesseract_lang("xyz") == "xyz"

    def test_already_tesseract_code_passes_through(self) -> None:
        assert _to_tesseract_lang("eng") == "eng"


# ---------------------------------------------------------------------------
# TestPaddleOCRLanguageMapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaddleOCRLanguageMapping:
    """Tests for _to_paddleocr_lang() and _PADDLEOCR_LANG_MAP."""

    def test_chinese_maps_to_ch(self) -> None:
        assert _to_paddleocr_lang("zh") == "ch"

    def test_korean_maps_to_korean(self) -> None:
        assert _to_paddleocr_lang("ko") == "korean"

    def test_japanese_maps_to_japan(self) -> None:
        assert _to_paddleocr_lang("ja") == "japan"

    def test_german_maps_to_german(self) -> None:
        assert _to_paddleocr_lang("de") == "german"

    def test_english_passes_through(self) -> None:
        assert _to_paddleocr_lang("en") == "en"

    def test_french_passes_through(self) -> None:
        assert _to_paddleocr_lang("fr") == "fr"

    def test_unknown_code_passes_through(self) -> None:
        assert _to_paddleocr_lang("xyz") == "xyz"

    def test_map_only_contains_exceptions(self) -> None:
        """Map should only contain codes where PaddleOCR deviates from ISO 639-1."""
        assert "en" not in _PADDLEOCR_LANG_MAP
        assert "fr" not in _PADDLEOCR_LANG_MAP


# ---------------------------------------------------------------------------
# TestTesseractAvailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTesseractAvailable:
    """Tests for _tesseract_available() helper."""

    @patch("ingestkit_pdf.utils.ocr_engines.shutil.which", return_value="/usr/bin/tesseract")
    def test_both_binary_and_module_available(self, mock_which: MagicMock) -> None:
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"pytesseract": mock_module}):
            assert _tesseract_available() is True

    @patch("ingestkit_pdf.utils.ocr_engines.shutil.which", return_value=None)
    def test_binary_missing(self, mock_which: MagicMock) -> None:
        assert _tesseract_available() is False

    @patch("ingestkit_pdf.utils.ocr_engines.shutil.which", return_value="/usr/bin/tesseract")
    def test_module_missing(self, mock_which: MagicMock) -> None:
        # Remove pytesseract from sys.modules so the import inside fails
        saved = sys.modules.pop("pytesseract", None)
        try:
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "pytesseract":
                    raise ImportError("mocked missing")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fake_import):
                assert _tesseract_available() is False
        finally:
            if saved is not None:
                sys.modules["pytesseract"] = saved

    @patch("ingestkit_pdf.utils.ocr_engines.shutil.which", return_value=None)
    def test_both_missing(self, mock_which: MagicMock) -> None:
        """Short-circuits on missing binary; never tries import."""
        assert _tesseract_available() is False


# ---------------------------------------------------------------------------
# TestTesseractEngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTesseractEngine:
    """Tests for TesseractEngine adapter."""

    @pytest.fixture()
    def mock_pytesseract(self) -> MagicMock:
        mock_module = MagicMock()
        mock_module.Output.DICT = "dict"
        with patch.dict(sys.modules, {"pytesseract": mock_module}):
            yield mock_module

    def test_name_returns_tesseract(self) -> None:
        assert TesseractEngine().name() == "tesseract"

    def test_recognize_basic(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["Hello", "beautiful", "world"],
            confs=[90, 80, 70],
        )
        engine = TesseractEngine()
        image = MagicMock()
        result = engine.recognize(image, language="en")

        assert result.text == "Hello beautiful world"
        assert result.confidence == pytest.approx(0.8)
        assert result.word_confidences == [0.9, 0.8, 0.7]

    def test_recognize_filters_empty_text(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["Hello", "", "  ", "world"],
            confs=[90, 50, 60, 80],
        )
        engine = TesseractEngine()
        result = engine.recognize(MagicMock(), language="en")

        assert result.text == "Hello world"
        assert len(result.word_confidences) == 2

    def test_recognize_filters_negative_confidence(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["Hello", "noise", "world"],
            confs=[90, -1, 80],
        )
        engine = TesseractEngine()
        result = engine.recognize(MagicMock(), language="en")

        assert result.text == "Hello world"
        assert len(result.word_confidences) == 2

    def test_recognize_empty_page(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["", "  ", ""],
            confs=[-1, -1, -1],
        )
        engine = TesseractEngine()
        result = engine.recognize(MagicMock(), language="en")

        assert result.text == ""
        assert result.confidence == 0.0
        assert result.word_confidences is None

    def test_recognize_maps_language(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["Bonjour"], confs=[95]
        )
        engine = TesseractEngine()
        engine.recognize(MagicMock(), language="fr")

        call_kwargs = mock_pytesseract.image_to_data.call_args
        assert call_kwargs[1]["lang"] == "fra"

    def test_confidence_scaled_to_0_1(self, mock_pytesseract: MagicMock) -> None:
        mock_pytesseract.image_to_data.return_value = _mock_tesseract_data(
            texts=["word"], confs=[95]
        )
        engine = TesseractEngine()
        result = engine.recognize(MagicMock(), language="en")

        assert result.word_confidences == [0.95]

    def test_default_lang_is_eng(self) -> None:
        engine = TesseractEngine()
        assert engine._lang == "eng"


# ---------------------------------------------------------------------------
# Helpers — PaddleOCR
# ---------------------------------------------------------------------------


def _mock_paddleocr_result(
    lines: list[tuple[str, float]],
) -> list[list[list]]:
    """Build a mock PaddleOCR ocr() return value.

    PaddleOCR returns: [[[bbox, (text, conf)], ...]]
    """
    page_lines = []
    for text, conf in lines:
        bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]  # dummy bbox
        page_lines.append([bbox, (text, conf)])
    return [page_lines]


# ---------------------------------------------------------------------------
# TestPaddleOCREngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPaddleOCREngine:
    """Tests for PaddleOCREngine adapter (mock-based, no real PaddleOCR)."""

    @pytest.fixture()
    def mock_paddleocr(self) -> MagicMock:
        """Mock the paddleocr module and its PaddleOCR class."""
        mock_module = MagicMock()
        mock_ocr_instance = MagicMock()
        mock_module.PaddleOCR.return_value = mock_ocr_instance
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            yield mock_module

    @pytest.fixture()
    def mock_numpy(self) -> MagicMock:
        """Mock numpy.array for PIL-to-ndarray conversion.

        We patch only numpy.array (not the whole module in sys.modules)
        because pytest.approx relies on the real numpy.bool_ type.
        """
        mock_np = MagicMock()
        mock_np.array.return_value = MagicMock()  # fake ndarray
        with patch("numpy.array", mock_np.array):
            yield mock_np

    def test_name_returns_paddleocr(self, mock_paddleocr: MagicMock) -> None:
        engine = PaddleOCREngine()
        assert engine.name() == "paddleocr"

    def test_init_creates_paddleocr_instance(self, mock_paddleocr: MagicMock) -> None:
        PaddleOCREngine(lang="en")
        mock_paddleocr.PaddleOCR.assert_called_once_with(
            lang="en", use_angle_cls=True, show_log=False,
        )

    def test_init_maps_language(self, mock_paddleocr: MagicMock) -> None:
        PaddleOCREngine(lang="zh")
        mock_paddleocr.PaddleOCR.assert_called_once_with(
            lang="ch", use_angle_cls=True, show_log=False,
        )

    def test_init_raises_import_error_when_not_installed(self) -> None:
        """Without paddleocr in sys.modules, __init__ raises ImportError."""
        saved = sys.modules.pop("paddleocr", None)
        try:
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __import__
            )

            def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "paddleocr":
                    raise ImportError("mocked missing")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fake_import):
                with pytest.raises(ImportError):
                    PaddleOCREngine()
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved

    def test_recognize_basic(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = _mock_paddleocr_result([
            ("Hello world", 0.95),
            ("Second line", 0.88),
        ])

        image = MagicMock()
        result = engine.recognize(image, language="en")

        assert result.text == "Hello world\nSecond line"
        assert result.confidence == pytest.approx(0.915)
        assert result.word_confidences == [0.95, 0.88]
        mock_numpy.array.assert_called_once_with(image)

    def test_recognize_empty_page_none_result(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = None

        result = engine.recognize(MagicMock(), language="en")

        assert result.text == ""
        assert result.confidence == 0.0
        assert result.word_confidences is None

    def test_recognize_empty_page_nested_none(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = [[None]]

        result = engine.recognize(MagicMock(), language="en")

        assert result.text == ""
        assert result.confidence == 0.0

    def test_recognize_empty_list_result(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = [[]]

        result = engine.recognize(MagicMock(), language="en")

        assert result.text == ""
        assert result.confidence == 0.0

    def test_recognize_filters_whitespace_only_lines(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = _mock_paddleocr_result([
            ("Hello", 0.9),
            ("  ", 0.5),
            ("World", 0.8),
        ])

        result = engine.recognize(MagicMock(), language="en")

        assert result.text == "Hello\nWorld"
        assert len(result.word_confidences) == 2

    def test_recognize_maps_language(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        """When recognize() language differs from init, a new OCR instance is created."""
        engine = PaddleOCREngine(lang="en")
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = _mock_paddleocr_result([("text", 0.9)])

        engine.recognize(MagicMock(), language="zh")

        # Second call to PaddleOCR() for the different language
        calls = mock_paddleocr.PaddleOCR.call_args_list
        assert len(calls) == 2
        assert calls[1][1]["lang"] == "ch"

    def test_confidence_already_0_to_1(
        self, mock_paddleocr: MagicMock, mock_numpy: MagicMock,
    ) -> None:
        """PaddleOCR confidence is already 0-1 (unlike Tesseract's 0-100)."""
        engine = PaddleOCREngine()
        ocr_instance = mock_paddleocr.PaddleOCR.return_value
        ocr_instance.ocr.return_value = _mock_paddleocr_result([
            ("word", 0.966),
        ])

        result = engine.recognize(MagicMock(), language="en")

        assert result.word_confidences == [0.966]
        assert result.confidence == pytest.approx(0.966)

    def test_satisfies_protocol(self, mock_paddleocr: MagicMock) -> None:
        engine = PaddleOCREngine()
        assert isinstance(engine, OCREngineInterface)

    def test_default_lang_is_en(self, mock_paddleocr: MagicMock) -> None:
        engine = PaddleOCREngine()
        assert engine._lang == "en"


# ---------------------------------------------------------------------------
# TestCreateOCREngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateOCREngine:
    """Tests for create_ocr_engine() factory."""

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_tesseract_config_returns_tesseract_engine(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.TESSERACT)
        engine, warnings = create_ocr_engine(config)
        assert isinstance(engine, TesseractEngine)
        assert warnings == []

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_tesseract_engine_receives_config_language(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.TESSERACT, ocr_language="fr")
        engine, _ = create_ocr_engine(config)
        assert engine._lang == "fra"

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_paddleocr_config_falls_back_to_tesseract(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.PADDLEOCR)
        # paddleocr is not installed in test env, so ImportError is natural
        saved = sys.modules.pop("paddleocr", None)
        try:
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "paddleocr":
                    raise ImportError("mocked missing")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fake_import):
                engine, warnings = create_ocr_engine(config)
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved

        assert isinstance(engine, TesseractEngine)
        assert "W_OCR_ENGINE_FALLBACK" in warnings

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_paddleocr_fallback_warning_message(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.PADDLEOCR)
        saved = sys.modules.pop("paddleocr", None)
        try:
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "paddleocr":
                    raise ImportError("mocked missing")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_fake_import):
                _, warnings = create_ocr_engine(config)
        finally:
            if saved is not None:
                sys.modules["paddleocr"] = saved

        assert len(warnings) == 1
        assert warnings[0] == "W_OCR_ENGINE_FALLBACK"

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=False)
    def test_tesseract_unavailable_raises_error(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.TESSERACT)
        with pytest.raises(EngineUnavailableError):
            create_ocr_engine(config)

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=False)
    def test_error_message_contains_install_instructions(
        self, mock_avail: MagicMock
    ) -> None:
        config = _make_config(ocr_engine=OCREngine.TESSERACT)
        with pytest.raises(EngineUnavailableError, match="apt install tesseract-ocr"):
            create_ocr_engine(config)
        with pytest.raises(EngineUnavailableError, match="pip install pytesseract"):
            create_ocr_engine(config)

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_paddleocr_config_returns_paddleocr_engine_when_available(
        self, mock_avail: MagicMock,
    ) -> None:
        mock_module = MagicMock()
        mock_module.PaddleOCR.return_value = MagicMock()
        with patch.dict(sys.modules, {"paddleocr": mock_module}):
            config = _make_config(ocr_engine=OCREngine.PADDLEOCR)
            engine, warnings = create_ocr_engine(config)
            assert isinstance(engine, PaddleOCREngine)
            assert warnings == []

    @patch("ingestkit_pdf.utils.ocr_engines._tesseract_available", return_value=True)
    def test_return_type_is_tuple(self, mock_avail: MagicMock) -> None:
        config = _make_config(ocr_engine=OCREngine.TESSERACT)
        result = create_ocr_engine(config)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# TestEngineUnavailableError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineUnavailableError:
    """Tests for EngineUnavailableError exception."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(EngineUnavailableError, Exception)

    def test_message_preserved(self) -> None:
        err = EngineUnavailableError("custom message")
        assert str(err) == "custom message"


# ---------------------------------------------------------------------------
# TestPaddleOCREngineIntegration
# ---------------------------------------------------------------------------


try:
    import paddleocr as _paddleocr_check  # noqa: F401

    _has_paddleocr = True
except ImportError:
    _has_paddleocr = False


@pytest.mark.ocr_paddle
@pytest.mark.skipif(not _has_paddleocr, reason="PaddleOCR not installed")
class TestPaddleOCREngineIntegration:
    """Integration tests requiring real PaddleOCR installation.

    Run with: pytest -m ocr_paddle
    Skip in CI unless PaddleOCR is installed.
    """

    def test_recognize_simple_image(self) -> None:
        """Smoke test: recognize text from a simple synthetic image."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (200, 50), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello", fill="black")

        engine = PaddleOCREngine(lang="en")
        result = engine.recognize(img, language="en")

        assert isinstance(result, OCRPageResult)
        assert result.confidence > 0.0
        assert len(result.text) > 0
