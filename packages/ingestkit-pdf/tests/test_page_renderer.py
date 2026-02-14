"""Tests for ingestkit_pdf.utils.page_renderer — PageRenderer."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.utils.page_renderer import PageRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_image(
    width: int = 200, height: int = 200, mode: str = "RGB"
) -> Image.Image:
    """Create a simple synthetic PIL Image for testing."""
    if mode == "RGB":
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        arr = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return Image.fromarray(arr, mode=mode)


def _make_mock_pixmap(width: int = 200, height: int = 200) -> MagicMock:
    """Create a mock fitz Pixmap with correct attributes."""
    pix = MagicMock()
    pix.width = width
    pix.height = height
    # 3 bytes per pixel (RGB)
    pix.samples = bytes(np.random.randint(0, 256, width * height * 3, dtype=np.uint8))
    return pix


def _make_mock_page(pixmap: MagicMock | None = None) -> MagicMock:
    """Create a mock fitz.Page that returns the given pixmap."""
    if pixmap is None:
        pixmap = _make_mock_pixmap()
    page = MagicMock()
    page.get_pixmap.return_value = pixmap
    return page


# ---------------------------------------------------------------------------
# TestRenderPage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderPage:
    def test_render_page_default_dpi(self):
        """Renders at default 300 DPI; resulting image matches pixmap dimensions."""
        config = PDFProcessorConfig()
        renderer = PageRenderer(config)

        pix = _make_mock_pixmap(width=2550, height=3300)
        page = _make_mock_page(pix)

        result = renderer.render_page(page)

        page.get_pixmap.assert_called_once_with(dpi=300)
        assert isinstance(result, Image.Image)
        assert result.size == (2550, 3300)
        assert result.mode == "RGB"

    def test_render_page_custom_dpi(self):
        """Renders at custom DPI; get_pixmap called with correct value."""
        config = PDFProcessorConfig(ocr_dpi=150)
        renderer = PageRenderer(config)

        pix = _make_mock_pixmap(width=1275, height=1650)
        page = _make_mock_page(pix)

        result = renderer.render_page(page)

        page.get_pixmap.assert_called_once_with(dpi=150)
        assert result.size == (1275, 1650)

    def test_render_page_large_dimensions(self, caplog):
        """Large pixmap dimensions trigger a warning but do not crash."""
        config = PDFProcessorConfig()
        renderer = PageRenderer(config)

        pix = _make_mock_pixmap(width=12000, height=8000)
        page = _make_mock_page(pix)

        with caplog.at_level(logging.WARNING, logger="ingestkit_pdf.utils.page_renderer"):
            result = renderer.render_page(page)

        assert result.size == (12000, 8000)
        assert "Large page dimensions" in caplog.text


# ---------------------------------------------------------------------------
# TestPreprocess
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPreprocessNoOpenCV:
    """Tests that do not require OpenCV."""

    def test_preprocess_no_steps(self):
        """Empty preprocessing steps returns image unchanged."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=[])
        renderer = PageRenderer(config)

        image = _make_synthetic_image(100, 100)
        result = renderer.preprocess(image)

        assert result is image  # same object, not a copy

    def test_preprocess_unknown_step_skipped(self, caplog):
        """Unknown step names are filtered at init with a warning."""
        with caplog.at_level(logging.WARNING, logger="ingestkit_pdf.utils.page_renderer"):
            config = PDFProcessorConfig(
                ocr_preprocessing_steps=["invalid_step", "another_bad"]
            )
            renderer = PageRenderer(config)

        assert "Unknown preprocessing step 'invalid_step'" in caplog.text
        assert "Unknown preprocessing step 'another_bad'" in caplog.text
        # No valid steps remain
        assert renderer._steps == []

    def test_unknown_mixed_with_valid(self, caplog):
        """Valid steps preserved, unknown steps logged and skipped."""
        with caplog.at_level(logging.WARNING, logger="ingestkit_pdf.utils.page_renderer"):
            config = PDFProcessorConfig(
                ocr_preprocessing_steps=["deskew", "invalid_step"]
            )
            renderer = PageRenderer(config)

        assert renderer._steps == ["deskew"]
        assert "Unknown preprocessing step 'invalid_step'" in caplog.text


@pytest.mark.unit
class TestPreprocessWithOpenCV:
    """Tests that require OpenCV (skip if not installed)."""

    cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")

    def test_preprocess_deskew(self):
        """Deskew step returns an image of the same dimensions."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["deskew"])
        renderer = PageRenderer(config)

        image = _make_synthetic_image(200, 200)
        result = renderer.preprocess(image)

        assert isinstance(result, Image.Image)
        assert result.size == image.size
        assert result.mode == "RGB"

    def test_preprocess_denoise(self):
        """Denoise step returns an image without error."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["denoise"])
        renderer = PageRenderer(config)

        image = _make_synthetic_image(200, 200)
        result = renderer.preprocess(image)

        assert isinstance(result, Image.Image)
        assert result.size == image.size

    def test_preprocess_binarize(self):
        """Binarize step produces output with only black and white pixels."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["binarize"])
        renderer = PageRenderer(config)

        # Create a gradient image for binarization
        gradient = np.tile(
            np.linspace(0, 255, 200, dtype=np.uint8), (200, 1)
        )
        rgb_gradient = np.stack([gradient, gradient, gradient], axis=-1)
        image = Image.fromarray(rgb_gradient, mode="RGB")

        result = renderer.preprocess(image)

        arr = np.array(result)
        unique_values = set(np.unique(arr))
        # After Otsu binarization, only 0 and 255 should remain
        assert unique_values <= {0, 255}

    def test_preprocess_contrast(self):
        """Contrast (CLAHE) step returns an image without error."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["contrast"])
        renderer = PageRenderer(config)

        # Create a low-contrast image
        low_contrast = np.full((200, 200, 3), 128, dtype=np.uint8)
        low_contrast[:100, :, :] = 120
        low_contrast[100:, :, :] = 136
        image = Image.fromarray(low_contrast, mode="RGB")

        result = renderer.preprocess(image)

        assert isinstance(result, Image.Image)
        assert result.size == image.size

    def test_preprocess_multiple_steps(self):
        """Multiple steps applied in configured order."""
        config = PDFProcessorConfig(
            ocr_preprocessing_steps=["deskew", "denoise", "binarize"]
        )
        renderer = PageRenderer(config)

        # Verify steps are stored in order
        assert renderer._steps == ["deskew", "denoise", "binarize"]

        # Mock internal methods to verify call order
        call_order: list[str] = []
        original_deskew = renderer._deskew
        original_denoise = renderer._denoise
        original_binarize = renderer._binarize

        def mock_deskew(arr):
            call_order.append("deskew")
            return original_deskew(arr)

        def mock_denoise(arr):
            call_order.append("denoise")
            return original_denoise(arr)

        def mock_binarize(arr):
            call_order.append("binarize")
            return original_binarize(arr)

        renderer._dispatch["deskew"] = mock_deskew
        renderer._dispatch["denoise"] = mock_denoise
        renderer._dispatch["binarize"] = mock_binarize

        image = _make_synthetic_image(200, 200)
        result = renderer.preprocess(image)

        assert call_order == ["deskew", "denoise", "binarize"]
        assert isinstance(result, Image.Image)

    def test_preprocess_grayscale_input(self):
        """Grayscale (mode 'L') input is converted to RGB and processed."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["denoise"])
        renderer = PageRenderer(config)

        gray_image = _make_synthetic_image(200, 200, mode="L")
        assert gray_image.mode == "L"

        result = renderer.preprocess(gray_image)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# TestOpenCVNotInstalled
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenCVNotInstalled:
    def test_cv2_import_error(self):
        """Clear error when cv2 is unavailable and preprocessing is attempted."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=["deskew"])
        renderer = PageRenderer(config)

        image = _make_synthetic_image(200, 200)

        with patch.dict("sys.modules", {"cv2": None}):
            with pytest.raises(ImportError, match="opencv-python-headless"):
                renderer.preprocess(image)


# ---------------------------------------------------------------------------
# TestRenderAndPreprocessPipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderAndPreprocessPipeline:
    def test_render_then_preprocess_end_to_end(self):
        """Full pipeline: render mock page, then preprocess with no steps."""
        config = PDFProcessorConfig(ocr_preprocessing_steps=[])
        renderer = PageRenderer(config)

        pix = _make_mock_pixmap(width=200, height=200)
        page = _make_mock_page(pix)

        rendered = renderer.render_page(page)
        result = renderer.preprocess(rendered)

        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)
