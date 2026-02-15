"""Tests for form field extractors — NativePDF and OCR overlay extraction."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.extractors._preprocessing import (
    adaptive_threshold,
    compute_fill_ratio,
    compute_ink_ratio,
    deskew,
    enhance_contrast,
    preprocess_for_ocr,
    reduce_noise,
)
from ingestkit_forms.extractors._rendering import (
    get_page_image,
    load_image_file,
    validate_image_safety,
)
from ingestkit_forms.extractors.native_pdf import NativePDFExtractor
from ingestkit_forms.extractors.ocr_overlay import (
    OCROverlayExtractor,
    _build_ocr_config,
    _crop_field_region,
    _post_process_value,
)
from ingestkit_forms.security import regex_match_with_timeout
from ingestkit_forms.models import (
    BoundingBox,
    CellAddress,
    ExtractedField,
    FieldMapping,
    FieldType,
    FormTemplate,
    SourceFormat,
)
from ingestkit_forms.protocols import OCRRegionResult, WidgetField


# ---------------------------------------------------------------------------
# Test-local mock backends and factories
# ---------------------------------------------------------------------------


class _MockPDFWidgetBackend:
    """Mock PDFWidgetBackend for testing NativePDFExtractor."""

    def __init__(
        self,
        widgets_by_page: dict[int, list[WidgetField]] | None = None,
        has_fields: bool = True,
    ) -> None:
        self._widgets_by_page = widgets_by_page or {}
        self._has_fields = has_fields

    def extract_widgets(self, file_path: str, page: int) -> list[WidgetField]:
        return self._widgets_by_page.get(page, [])

    def has_form_fields(self, file_path: str) -> bool:
        return self._has_fields

    def engine_name(self) -> str:
        return "mock"


class _MockOCRBackend:
    """Mock OCRBackend for testing OCR fallback paths."""

    def __init__(
        self, default_text: str = "", default_confidence: float = 0.7
    ) -> None:
        self._default_text = default_text
        self._default_confidence = default_confidence

    def ocr_region(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ) -> OCRRegionResult:
        return OCRRegionResult(
            text=self._default_text,
            confidence=self._default_confidence,
            char_confidences=None,
            engine="mock",
        )

    def engine_name(self) -> str:
        return "mock"


def _widget(
    field_name: str = "field_1",
    field_value: str | None = "some value",
    field_type: str = "text",
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.3,
    height: float = 0.05,
    page: int = 0,
) -> WidgetField:
    return WidgetField(
        field_name=field_name,
        field_value=field_value,
        field_type=field_type,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
        page=page,
    )


def _field(
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    page_number: int = 0,
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.3,
    height: float = 0.05,
    validation_pattern: str | None = None,
    required: bool = False,
    options: list[str] | None = None,
) -> FieldMapping:
    return FieldMapping(
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        page_number=page_number,
        region=BoundingBox(x=x, y=y, width=width, height=height),
        validation_pattern=validation_pattern,
        required=required,
        options=options,
    )


def _template(
    fields: list[FieldMapping] | None = None,
    name: str = "Test Template",
    source_format: SourceFormat = SourceFormat.PDF,
    page_count: int = 1,
) -> FormTemplate:
    if fields is None:
        fields = [_field()]
    return FormTemplate(
        name=name,
        source_format=source_format,
        page_count=page_count,
        fields=fields,
    )


# ---------------------------------------------------------------------------
# Test 1: Text field with exact bbox match
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_text_field_exact_match(form_config):
    widget = _widget(
        field_name="name", field_value="John Doe",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_name="employee_name",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert len(results) == 1
    assert results[0].value == "John Doe"
    assert results[0].confidence == 0.99  # native_fields, no coercion -> 0.99
    assert results[0].extraction_method == "native_fields"
    assert results[0].field_name == "employee_name"


# ---------------------------------------------------------------------------
# Test 2: Text field with IoU above threshold (offset bbox)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_text_field_iou_above_threshold(form_config):
    widget = _widget(
        field_value="Jane Smith",
        x=0.12, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(x=0.1, y=0.1, width=0.3, height=0.05)
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value == "Jane Smith"
    assert results[0].confidence == 0.99  # native_fields, no coercion -> 0.99


# ---------------------------------------------------------------------------
# Test 3: Text field with IoU below threshold (no overlap)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_text_field_iou_below_threshold(form_config):
    widget = _widget(
        field_value="Won't Match",
        x=0.7, y=0.7, width=0.2, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(x=0.1, y=0.1, width=0.3, height=0.05)
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value is None
    assert results[0].confidence == 0.0
    assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in results[0].warnings


# ---------------------------------------------------------------------------
# Test 4: Checkbox checked (value "Yes" -> True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_checkbox_checked(form_config):
    widget = _widget(
        field_value="Yes", field_type="checkbox",
        x=0.1, y=0.2, width=0.05, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_type=FieldType.CHECKBOX,
        x=0.1, y=0.2, width=0.05, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value is True
    assert results[0].field_type == FieldType.CHECKBOX


# ---------------------------------------------------------------------------
# Test 5: Checkbox unchecked (value "Off" -> False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_checkbox_unchecked(form_config):
    widget = _widget(
        field_value="Off", field_type="checkbox",
        x=0.1, y=0.2, width=0.05, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_type=FieldType.CHECKBOX,
        x=0.1, y=0.2, width=0.05, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value is False


# ---------------------------------------------------------------------------
# Test 6: Radio button (value "Yes" -> True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_radio_button(form_config):
    widget = _widget(
        field_value="Yes", field_type="radio",
        x=0.1, y=0.3, width=0.05, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_type=FieldType.RADIO,
        x=0.1, y=0.3, width=0.05, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value is True
    assert results[0].field_type == FieldType.RADIO


# ---------------------------------------------------------------------------
# Test 7: Dropdown (string value returned)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_dropdown(form_config):
    widget = _widget(
        field_value="Full-Time", field_type="dropdown",
        x=0.1, y=0.4, width=0.2, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_type=FieldType.DROPDOWN,
        x=0.1, y=0.4, width=0.2, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value == "Full-Time"
    assert results[0].field_type == FieldType.DROPDOWN


# ---------------------------------------------------------------------------
# Test 8: Number field coercion (string "42.5" -> float 42.5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_number_field_coercion(form_config):
    widget = _widget(
        field_value="42.5", field_type="text",
        x=0.1, y=0.5, width=0.15, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        field_type=FieldType.NUMBER,
        x=0.1, y=0.5, width=0.15, height=0.05,
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value == 42.5
    assert isinstance(results[0].value, float)


# ---------------------------------------------------------------------------
# Test 9: Flattened form detection (has_form_fields returns False)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flattened_form_detection(form_config):
    backend = _MockPDFWidgetBackend(has_fields=False)
    template = _template()

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results == []


# ---------------------------------------------------------------------------
# Test 10: Validation pattern pass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validation_pattern_pass(form_config):
    widget = _widget(
        field_value="123-45-6789",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        x=0.1, y=0.1, width=0.3, height=0.05,
        validation_pattern=r"^\d{3}-\d{2}-\d{4}$",
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].validation_passed is True
    assert results[0].warnings == []
    assert results[0].value == "123-45-6789"


# ---------------------------------------------------------------------------
# Test 11: Validation pattern fail
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validation_pattern_fail(form_config):
    widget = _widget(
        field_value="not-a-ssn",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(
        x=0.1, y=0.1, width=0.3, height=0.05,
        validation_pattern=r"^\d{3}-\d{2}-\d{4}$",
    )
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].validation_passed is False
    assert FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value in results[0].warnings


# ---------------------------------------------------------------------------
# Test 12: Multiple fields across multiple pages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multiple_fields_multiple_pages(form_config):
    widget_p0 = _widget(
        field_name="w_name", field_value="Alice",
        x=0.1, y=0.1, width=0.3, height=0.05, page=0,
    )
    widget_p1 = _widget(
        field_name="w_dept", field_value="Engineering",
        x=0.1, y=0.3, width=0.3, height=0.05, page=1,
    )
    backend = _MockPDFWidgetBackend(
        widgets_by_page={0: [widget_p0], 1: [widget_p1]}
    )
    field_p0 = _field(
        field_name="name", page_number=0,
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    field_p1 = _field(
        field_name="department", page_number=1,
        x=0.1, y=0.3, width=0.3, height=0.05,
    )
    template = _template(fields=[field_p0, field_p1], page_count=2)

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert len(results) == 2
    assert results[0].value == "Alice"
    assert results[0].field_name == "name"
    assert results[1].value == "Engineering"
    assert results[1].field_name == "department"


# ---------------------------------------------------------------------------
# Test 13: No OCR backend + unmatched field -> value=None, method="native_fields"
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_ocr_backend_unmatched_field(form_config):
    widget = _widget(
        field_value="Far Away", x=0.7, y=0.7, width=0.2, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(x=0.1, y=0.1, width=0.3, height=0.05)
    template = _template(fields=[fld])

    extractor = NativePDFExtractor(pdf_backend=backend, config=form_config)
    results = extractor.extract("/fake/form.pdf", template)

    assert results[0].value is None
    assert results[0].extraction_method == "native_fields"


# ---------------------------------------------------------------------------
# Test 14: Configurable IoU threshold
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_configurable_iou_threshold():
    # Widget slightly offset: IoU ~ 0.714 (above 0.5, below 0.8)
    widget = _widget(
        field_value="Test", x=0.15, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})
    fld = _field(x=0.1, y=0.1, width=0.3, height=0.05)
    template = _template(fields=[fld])

    # Default threshold (0.5) -> match
    config_low = FormProcessorConfig()
    extractor_low = NativePDFExtractor(pdf_backend=backend, config=config_low)
    results_low = extractor_low.extract("/fake/form.pdf", template)
    assert results_low[0].value == "Test"

    # High threshold (0.8) -> no match
    config_high = FormProcessorConfig(native_pdf_iou_threshold=0.8)
    extractor_high = NativePDFExtractor(pdf_backend=backend, config=config_high)
    results_high = extractor_high.extract("/fake/form.pdf", template)
    assert results_high[0].value is None


# ---------------------------------------------------------------------------
# Test 15: Direct IoU computation unit test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iou_computation_direct():
    # Identical boxes -> IoU = 1.0
    box = BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3)
    assert abs(NativePDFExtractor._compute_iou(box, box) - 1.0) < 1e-9

    # No overlap -> IoU = 0.0
    box_a = BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)
    box_b = BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1)
    assert abs(NativePDFExtractor._compute_iou(box_a, box_b) - 0.0) < 1e-9

    # 50% overlap in one dimension
    box_c = BoundingBox(x=0.0, y=0.0, width=0.2, height=0.1)
    box_d = BoundingBox(x=0.1, y=0.0, width=0.2, height=0.1)
    # Intersection: (0.1, 0.0) to (0.2, 0.1) = 0.1 * 0.1 = 0.01
    # Union: 0.02 + 0.02 - 0.01 = 0.03
    # IoU = 0.01 / 0.03 = 0.333...
    iou = NativePDFExtractor._compute_iou(box_c, box_d)
    assert abs(iou - 1.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 16: Extraction method labels (matched vs OCR fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extraction_method_label(form_config):
    widget = _widget(
        field_name="w1", field_value="Matched",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    backend = _MockPDFWidgetBackend(widgets_by_page={0: [widget]})

    field_matched = _field(
        field_name="field_1",
        x=0.1, y=0.1, width=0.3, height=0.05,
    )
    field_unmatched = _field(
        field_name="field_2",
        x=0.7, y=0.7, width=0.2, height=0.05,
    )
    template = _template(fields=[field_matched, field_unmatched])

    ocr_backend = _MockOCRBackend()
    extractor = NativePDFExtractor(
        pdf_backend=backend, config=form_config, ocr_backend=ocr_backend,
    )
    results = extractor.extract("/fake/form.pdf", template)

    result_map = {r.field_name: r for r in results}
    assert result_map["field_1"].extraction_method == "native_fields"
    assert result_map["field_2"].extraction_method == "native_fields_with_ocr_fallback"


# ===========================================================================
# OCR Overlay Extraction Tests (issue #64)
# ===========================================================================


# ---------------------------------------------------------------------------
# OCR overlay helpers
# ---------------------------------------------------------------------------


def _make_white_image(width: int = 200, height: int = 50) -> Image.Image:
    """Create an all-white test image."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _make_black_image(width: int = 200, height: int = 50) -> Image.Image:
    """Create an all-black test image."""
    return Image.new("RGB", (width, height), color=(0, 0, 0))


def _make_half_filled_image(width: int = 200, height: int = 50) -> Image.Image:
    """Create image with top half black, bottom half white."""
    img = Image.new("L", (width, height), 255)
    arr = np.array(img)
    arr[: height // 2, :] = 0
    return Image.fromarray(arr)


def _make_low_contrast_image(width: int = 200, height: int = 50) -> Image.Image:
    """Create a low-contrast grayscale image (values 100-150)."""
    arr = np.random.default_rng(42).integers(
        100, 150, size=(height, width), dtype=np.uint8
    )
    return Image.fromarray(arr, mode="L")


def _save_image_to_tmp(img: Image.Image, tmp_path, name: str = "test.png") -> str:
    """Save a PIL image to a temp file and return the path."""
    path = tmp_path / name
    img.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# OCR overlay fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ocr_mock_backend():
    """Create a mock OCRBackend with MagicMock for call inspection."""
    backend = MagicMock()
    backend.engine_name.return_value = "tesseract"
    backend.ocr_region.return_value = OCRRegionResult(
        text="Sample Text",
        confidence=0.92,
        char_confidences=[
            0.9, 0.95, 0.88, 0.93, 0.94, 0.9, 0.91, 0.92, 0.95, 0.93, 0.89,
        ],
        engine="tesseract",
    )
    return backend


@pytest.fixture
def ocr_config():
    """Create a default FormProcessorConfig for OCR tests."""
    return FormProcessorConfig()


@pytest.fixture
def ocr_simple_template():
    """Create a single-page template with one text field."""
    return FormTemplate(
        name="Test Form",
        source_format=SourceFormat.PDF,
        page_count=1,
        fields=[
            FieldMapping(
                field_name="name",
                field_label="Full Name",
                field_type=FieldType.TEXT,
                page_number=0,
                region=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeskew:
    def test_deskew_returns_image(self):
        """Deskew a straight image returns same-size image."""
        img = _make_white_image(200, 100)
        result = deskew(img)
        assert isinstance(result, Image.Image)
        assert result.size == img.size

    def test_deskew_small_image_unchanged(self):
        """Image < 20x20 is returned unchanged."""
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))
        result = deskew(img)
        assert list(result.getdata()) == list(img.getdata())


@pytest.mark.unit
class TestEnhanceContrast:
    def test_enhance_contrast_increases_range(self):
        """Low-contrast image -> wider histogram spread."""
        img = _make_low_contrast_image(100, 100)
        result = enhance_contrast(img)
        result_arr = np.asarray(result)
        input_arr = np.asarray(img.convert("L"))
        assert (result_arr.max() - result_arr.min()) >= (
            input_arr.max() - input_arr.min()
        )

    def test_enhance_contrast_fallback_on_error(self):
        """Small images still return a valid image via fallback."""
        img = _make_white_image(30, 30)
        result = enhance_contrast(img)
        assert isinstance(result, Image.Image)


@pytest.mark.unit
class TestReduceNoise:
    def test_reduce_noise_preserves_size(self):
        """Output has same dimensions as input."""
        img = _make_white_image(100, 50)
        result = reduce_noise(img)
        assert result.size == (100, 50)


@pytest.mark.unit
class TestAdaptiveThreshold:
    def test_adaptive_threshold_produces_binary(self):
        """Grayscale input -> binary output (only 0 and 255 values)."""
        img = _make_low_contrast_image(100, 100)
        result = adaptive_threshold(img)
        arr = np.asarray(result)
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255})


@pytest.mark.unit
class TestPreprocessForOCR:
    def test_preprocess_for_ocr_text_pipeline(self):
        """TEXT field goes through full 4-step pipeline, returns an image."""
        img = _make_white_image(100, 50)
        result = preprocess_for_ocr(img, FieldType.TEXT)
        assert isinstance(result, Image.Image)

    def test_preprocess_for_ocr_checkbox_pipeline(self):
        """CHECKBOX field gets threshold only, returns a binary image."""
        img = _make_white_image(100, 50)
        result = preprocess_for_ocr(img, FieldType.CHECKBOX)
        arr = np.asarray(result.convert("L"))
        unique = set(np.unique(arr))
        assert unique.issubset({0, 255})


@pytest.mark.unit
class TestFillRatio:
    def test_compute_fill_ratio_empty_region(self):
        """All-white image -> ratio near 0.0."""
        img = _make_white_image()
        ratio = compute_fill_ratio(img)
        assert ratio < 0.01

    def test_compute_fill_ratio_filled_region(self):
        """All-black image -> ratio near 1.0."""
        img = _make_black_image()
        ratio = compute_fill_ratio(img)
        assert ratio > 0.99

    def test_compute_fill_ratio_partial(self):
        """Half-filled image -> ratio ~0.5."""
        img = _make_half_filled_image()
        ratio = compute_fill_ratio(img)
        assert 0.45 < ratio < 0.55

    def test_compute_ink_ratio_delegates(self):
        """compute_ink_ratio returns same as compute_fill_ratio."""
        img = _make_half_filled_image()
        assert compute_ink_ratio(img) == compute_fill_ratio(img)


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRendering:
    def test_load_image_file_jpeg(self, tmp_path):
        """Load a valid JPEG from tmp_path, returns RGB PIL Image."""
        img = _make_white_image(100, 100)
        path = _save_image_to_tmp(img, tmp_path, "test.jpg")
        result = load_image_file(path)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_load_image_file_rejects_oversized(self, tmp_path):
        """Image > max_dimension -> FormIngestException."""
        img = _make_white_image(100, 100)
        path = _save_image_to_tmp(img, tmp_path)
        with pytest.raises(FormIngestException) as exc_info:
            validate_image_safety(path, max_dimension=50)
        assert exc_info.value.code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_validate_image_safety_empty_file(self, tmp_path):
        """0-byte file -> FormIngestException."""
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(FormIngestException) as exc_info:
            validate_image_safety(str(path))
        assert exc_info.value.code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_render_pdf_page_import_error(self):
        """Mock fitz import failure -> FormIngestException."""
        with patch.dict("sys.modules", {"fitz": None}):
            with pytest.raises(FormIngestException) as exc_info:
                from ingestkit_forms.extractors._rendering import render_pdf_page

                render_pdf_page("dummy.pdf", 0)
            assert exc_info.value.code == FormErrorCode.E_FORM_UNSUPPORTED_FORMAT

    def test_get_page_image_unsupported_ext(self):
        """.docx file -> FormIngestException."""
        with pytest.raises(FormIngestException) as exc_info:
            get_page_image("test.docx", 0)
        assert exc_info.value.code == FormErrorCode.E_FORM_UNSUPPORTED_FORMAT

    def test_get_page_image_image_nonzero_page(self, tmp_path):
        """Image with page=1 -> FormIngestException."""
        img = _make_white_image(100, 100)
        path = _save_image_to_tmp(img, tmp_path, "test.png")
        with pytest.raises(FormIngestException) as exc_info:
            get_page_image(path, 1)
        assert exc_info.value.code == FormErrorCode.E_FORM_EXTRACTION_FAILED


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Text Fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlayText:
    def test_extract_text_field(
        self, ocr_mock_backend, ocr_config, ocr_simple_template, tmp_path
    ):
        """Mock OCR returns text, verify ExtractedField."""
        img = _make_white_image(1000, 800)
        path = _save_image_to_tmp(img, tmp_path, "test.png")
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract(path, ocr_simple_template)

        assert len(results) == 1
        field = results[0]
        assert field.value == "Sample Text"
        assert field.extraction_method == "ocr_overlay"
        assert field.confidence > 0.0

    def test_extract_number_field_strips_nonnumeric(self, ocr_config, tmp_path):
        """OCR returns '12.34 USD', verify '12.34'."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        backend.ocr_region.return_value = OCRRegionResult(
            text="12.34 USD", confidence=0.9, char_confidences=None, engine="tesseract",
        )
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="amount",
                    field_label="Amount",
                    field_type=FieldType.NUMBER,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value == "12.34"

    def test_extract_date_field(self, ocr_config):
        """OCR returns '02/15/2026', verify value."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        backend.ocr_region.return_value = OCRRegionResult(
            text="02/15/2026", confidence=0.95, char_confidences=None, engine="tesseract",
        )
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="date",
                    field_label="Date",
                    field_type=FieldType.DATE,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value == "02/15/2026"


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Checkbox
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlayCheckbox:
    def test_extract_checkbox_checked(self, ocr_mock_backend, ocr_config):
        """Mostly-dark crop (fill > 0.3) -> value=True.

        Uses a checkerboard-like image: dark center with white border so that
        adaptive thresholding preserves the dark region (local contrast exists).
        """
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="agree",
                    field_label="I Agree",
                    field_type=FieldType.CHECKBOX,
                    page_number=0,
                    region=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                ),
            ],
        )
        # Create an image with alternating dark and white horizontal stripes.
        # Adaptive thresholding preserves the dark stripes due to local contrast.
        # ~50% fill ratio after thresholding.
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        for row in range(0, 100, 4):
            arr[row : row + 2, :, :] = 0  # 2px dark, 2px white alternating
        img = Image.fromarray(arr)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value is True

    def test_extract_checkbox_unchecked(self, ocr_mock_backend, ocr_config):
        """All-white crop (fill < 0.3) -> value=False."""
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="agree",
                    field_label="I Agree",
                    field_type=FieldType.CHECKBOX,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.2, width=0.05, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value is False

    def test_extract_checkbox_confidence_calculation(self, ocr_mock_backend, ocr_config):
        """Confidence = min(abs(fill_ratio - 0.3) / 0.3, 1.0)."""
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="agree",
                    field_label="I Agree",
                    field_type=FieldType.CHECKBOX,
                    page_number=0,
                    region=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                ),
            ],
        )
        # All-black: fill_ratio ~1.0, confidence = min(abs(1.0-0.3)/0.3, 1.0) = 1.0
        img = _make_black_image(100, 100)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].confidence == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Signature
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlaySignature:
    def test_extract_signature_signed(self, ocr_mock_backend, ocr_config):
        """Crop with ink_ratio > 0.05 -> value=True.

        Uses scattered dark marks on a white background. The local contrast
        ensures adaptive thresholding preserves the dark ink regions.
        """
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="signature",
                    field_label="Signature",
                    field_type=FieldType.SIGNATURE,
                    page_number=0,
                    region=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                ),
            ],
        )
        # White background with dark "signature strokes" in the middle.
        # Adaptive threshold preserves these because of local contrast.
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Draw thick dark lines to simulate signature strokes (>5% ink)
        arr[30:35, 20:80, :] = 0   # horizontal stroke
        arr[40:45, 15:85, :] = 0   # another horizontal stroke
        arr[50:55, 25:75, :] = 0   # another
        arr[30:55, 45:50, :] = 0   # vertical stroke
        img = Image.fromarray(arr)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value is True

    def test_extract_signature_blank(self, ocr_mock_backend, ocr_config):
        """All-white crop -> value=False."""
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="signature",
                    field_label="Signature",
                    field_type=FieldType.SIGNATURE,
                    page_number=0,
                    region=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
                ),
            ],
        )
        img = _make_white_image(100, 100)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].value is False


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlayValidation:
    def test_validation_pattern_pass_ocr(self, ocr_config):
        """Regex matches -> validation_passed=True, value kept."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        backend.ocr_region.return_value = OCRRegionResult(
            text="123-4567", confidence=0.95, char_confidences=None, engine="tesseract",
        )
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="phone",
                    field_label="Phone",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05),
                    validation_pattern=r"^\d{3}-\d{4}$",
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].validation_passed is True
        assert results[0].value == "123-4567"

    def test_validation_pattern_fail_ocr(self, ocr_config):
        """Regex fails -> value=None, confidence=0.0, warning added."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        backend.ocr_region.return_value = OCRRegionResult(
            text="abc", confidence=0.95, char_confidences=None, engine="tesseract",
        )
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="code",
                    field_label="Code",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05),
                    validation_pattern=r"^\d{3}$",
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert results[0].validation_passed is False
        assert results[0].value is None
        assert results[0].confidence == 0.0
        assert FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value in results[0].warnings


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Timeout and Errors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlayTimeoutAndErrors:
    def test_per_field_timeout_passed_to_backend(
        self, ocr_mock_backend, ocr_config, ocr_simple_template
    ):
        """Verify timeout kwarg propagated to ocr_region."""
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            extractor.extract("test.png", ocr_simple_template)

        ocr_mock_backend.ocr_region.assert_called_once()
        call_kwargs = ocr_mock_backend.ocr_region.call_args
        # Check timeout was passed (either as kwarg or positional)
        timeout_val = call_kwargs.kwargs.get("timeout")
        assert timeout_val == float(ocr_config.form_ocr_per_field_timeout_seconds)

    def test_ocr_timeout_returns_none_value(self, ocr_config, ocr_simple_template):
        """Backend raises TimeoutError -> value=None, confidence=0.0."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        backend.ocr_region.side_effect = TimeoutError("OCR timed out")

        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", ocr_simple_template)

        assert results[0].value is None
        assert results[0].confidence == 0.0

    def test_ocr_failure_continues_to_next_field(self, ocr_config):
        """First field raises, second succeeds -> both in results."""
        backend = MagicMock()
        backend.engine_name.return_value = "tesseract"
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("OCR engine error")
            return OCRRegionResult(
                text="OK", confidence=0.95, char_confidences=None, engine="tesseract",
            )

        backend.ocr_region.side_effect = side_effect

        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="field1",
                    field_label="Field 1",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                ),
                FieldMapping(
                    field_name="field2",
                    field_label="Field 2",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.3, width=0.3, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.png", template)

        assert len(results) == 2
        assert results[0].value is None
        assert any(FormErrorCode.E_FORM_OCR_FAILED.value in w for w in results[0].warnings)
        assert results[1].value == "OK"

    def test_multi_page_extraction(self, ocr_mock_backend, ocr_config):
        """Fields on pages 0 and 1 -> get_page_image called twice."""
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=2,
            fields=[
                FieldMapping(
                    field_name="field_p0",
                    field_label="Page 0 Field",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                ),
                FieldMapping(
                    field_name="field_p1",
                    field_label="Page 1 Field",
                    field_type=FieldType.TEXT,
                    page_number=1,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ) as mock_get_page:
            results = extractor.extract("test.pdf", template)

        assert len(results) == 2
        assert mock_get_page.call_count == 2
        page_args = [call.args[1] for call in mock_get_page.call_args_list]
        assert 0 in page_args
        assert 1 in page_args


# ---------------------------------------------------------------------------
# OCROverlayExtractor Tests: Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCROverlayEdgeCases:
    def test_bbox_to_pixel_conversion(self):
        """1000x800 image, bbox(0.1, 0.2, 0.3, 0.05) -> crop at expected pixels."""
        img = Image.new("RGB", (1000, 800), color=(128, 128, 128))
        region = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.05)
        crop = _crop_field_region(img, region)
        # left=100, top=160, right=100+300=400, bottom=160+40=200
        assert crop.size == (300, 40)

    def test_field_without_region_fails_closed(self, ocr_mock_backend, ocr_config):
        """Field with cell_address (no region) -> value=None, error warning.

        PLAN-CHECK correction: FieldMapping requires one of region/cell_address.
        Use cell_address-based field to simulate an Excel field reaching OCR.
        """
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.XLSX,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="excel_field",
                    field_label="Excel Field",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    cell_address=CellAddress(cell="B2"),
                ),
            ],
        )
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            return_value=img,
        ):
            results = extractor.extract("test.xlsx", template)

        assert len(results) == 1
        assert results[0].value is None
        assert results[0].confidence == 0.0
        assert any(
            FormErrorCode.E_FORM_EXTRACTION_FAILED.value in w for w in results[0].warnings
        )

    def test_page_render_failure_skips_page_fields(self, ocr_mock_backend, ocr_config):
        """get_page_image raises for page 1 -> all page-1 fields have value=None."""
        template = FormTemplate(
            name="Test",
            source_format=SourceFormat.PDF,
            page_count=2,
            fields=[
                FieldMapping(
                    field_name="field_p0",
                    field_label="Page 0 Field",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                ),
                FieldMapping(
                    field_name="field_p1a",
                    field_label="Page 1 Field A",
                    field_type=FieldType.TEXT,
                    page_number=1,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                ),
                FieldMapping(
                    field_name="field_p1b",
                    field_label="Page 1 Field B",
                    field_type=FieldType.TEXT,
                    page_number=1,
                    region=BoundingBox(x=0.1, y=0.3, width=0.3, height=0.05),
                ),
            ],
        )
        img = _make_white_image(1000, 800)

        def mock_get_page(file_path, page, dpi=300):
            if page == 1:
                raise FormIngestException(
                    code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                    message="Page 1 rendering failed",
                    stage="rendering",
                    recoverable=False,
                )
            return img

        extractor = OCROverlayExtractor(ocr_mock_backend, ocr_config)

        with patch(
            "ingestkit_forms.extractors.ocr_overlay.get_page_image",
            side_effect=mock_get_page,
        ):
            results = extractor.extract("test.pdf", template)

        assert len(results) == 3
        assert results[0].value is not None  # Page 0 succeeded
        assert results[1].value is None  # Page 1 failed
        assert results[2].value is None
        assert any(
            FormErrorCode.E_FORM_EXTRACTION_FAILED.value in w for w in results[1].warnings
        )


# ---------------------------------------------------------------------------
# Security Tests (OCR overlay)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCRSecurity:
    def test_decompression_bomb_rejected(self, tmp_path):
        """High decompression ratio or oversized image -> FormIngestException."""
        img = _make_white_image(100, 100)
        path = _save_image_to_tmp(img, tmp_path, "bomb.png")
        # Use a very low dimension limit to trigger the security check
        with pytest.raises(FormIngestException) as exc_info:
            validate_image_safety(path, max_dimension=50)
        assert exc_info.value.code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_resolution_guardrail(self, tmp_path):
        """Image exceeding dimension limit -> rejected."""
        img = _make_white_image(200, 200)
        path = _save_image_to_tmp(img, tmp_path, "large.png")
        with pytest.raises(FormIngestException) as exc_info:
            validate_image_safety(path, max_dimension=100)
        assert exc_info.value.code == FormErrorCode.E_FORM_FILE_CORRUPT

    def test_regex_timeout_protection(self):
        """ReDoS pattern with pathological input -> returns None (timeout) or False."""
        result = regex_match_with_timeout(
            r"(a+)+$",
            "a" * 25 + "!",
            timeout=1.0,
            match_mode="match",
        )
        # Either times out (None) or fails to match (False)
        assert result is None or result is False


# ---------------------------------------------------------------------------
# PII Safety Tests (OCR overlay)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCRPIISafety:
    def test_no_raw_ocr_in_logs_by_default(
        self, ocr_mock_backend, ocr_simple_template, caplog
    ):
        """log_ocr_output=False -> no OCR text in log output."""
        config = FormProcessorConfig(log_ocr_output=False)
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, config)

        with (
            patch(
                "ingestkit_forms.extractors.ocr_overlay.get_page_image",
                return_value=img,
            ),
            caplog.at_level(logging.DEBUG, logger="ingestkit_forms"),
        ):
            extractor.extract("test.png", ocr_simple_template)

        assert "Sample Text" not in caplog.text

    def test_raw_ocr_logged_when_enabled(
        self, ocr_mock_backend, ocr_simple_template, caplog
    ):
        """log_ocr_output=True -> OCR text appears in log output."""
        config = FormProcessorConfig(log_ocr_output=True)
        img = _make_white_image(1000, 800)
        extractor = OCROverlayExtractor(ocr_mock_backend, config)

        with (
            patch(
                "ingestkit_forms.extractors.ocr_overlay.get_page_image",
                return_value=img,
            ),
            caplog.at_level(logging.DEBUG, logger="ingestkit_forms"),
        ):
            extractor.extract("test.png", ocr_simple_template)

        assert "Sample Text" in caplog.text


# ---------------------------------------------------------------------------
# Helper Function Tests (OCR overlay)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOCRHelperFunctions:
    def test_build_ocr_config_tesseract_number(self):
        """Tesseract + NUMBER -> whitelist config."""
        result = _build_ocr_config(FieldType.NUMBER, "tesseract")
        assert result == "--psm 7 -c tessedit_char_whitelist=0123456789.-"

    def test_build_ocr_config_tesseract_text(self):
        """Tesseract + TEXT -> '--psm 7'."""
        result = _build_ocr_config(FieldType.TEXT, "tesseract")
        assert result == "--psm 7"

    def test_build_ocr_config_paddleocr_number(self):
        """PaddleOCR + NUMBER -> 'rec_char_type=EN'."""
        result = _build_ocr_config(FieldType.NUMBER, "paddleocr")
        assert result == "rec_char_type=EN"

    def test_post_process_number_strips_text(self):
        """Input '$ 1,234.56 USD' -> '1,234.56'."""
        result = _post_process_value("$ 1,234.56 USD", FieldType.NUMBER)
        assert result == "1,234.56"


# ===========================================================================
# Excel Cell Extraction Tests (issue #65)
# ===========================================================================

from datetime import datetime
from typing import Any

from ingestkit_forms.extractors.excel_cell import ExcelCellExtractor


# ---------------------------------------------------------------------------
# Excel cell test helpers
# ---------------------------------------------------------------------------


def _excel_field(
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    cell: str = "B2",
    sheet_name: str | None = None,
    validation_pattern: str | None = None,
    required: bool = False,
    default_value: str | None = None,
    options: list[str] | None = None,
) -> FieldMapping:
    """Factory for FieldMapping with cell_address (Excel fields)."""
    return FieldMapping(
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        page_number=0,
        cell_address=CellAddress(cell=cell, sheet_name=sheet_name),
        validation_pattern=validation_pattern,
        required=required,
        default_value=default_value,
        options=options,
    )


def _excel_template(
    fields: list[FieldMapping] | None = None,
    name: str = "Excel Test Template",
) -> FormTemplate:
    """Factory for FormTemplate with Excel fields."""
    if fields is None:
        fields = [_excel_field()]
    return FormTemplate(
        name=name,
        source_format=SourceFormat.XLSX,
        page_count=1,
        fields=fields,
    )


def _mock_cell(value: Any = None) -> MagicMock:
    """Create a mock openpyxl cell with a value."""
    cell_mock = MagicMock()
    cell_mock.value = value
    return cell_mock


def _mock_worksheet(
    cells: dict[str, Any] | None = None,
    merged_ranges: list | None = None,
    range_data: dict[str, list[list[Any]]] | None = None,
) -> MagicMock:
    """Create a mock openpyxl worksheet.

    Args:
        cells: Mapping of cell address -> value, e.g. {"B2": "John Doe"}.
        merged_ranges: List of mock MergedCellRange objects.
        range_data: Mapping of range string -> list of rows of values.

    Note: MagicMock dunder methods pass ``self`` as first arg.
    """
    ws = MagicMock()

    if cells:
        def getitem(_self, key):
            if key in (range_data or {}):
                rows = range_data[key]
                result = []
                for row_vals in rows:
                    row_cells = []
                    for val in row_vals:
                        row_cells.append(_mock_cell(val))
                    result.append(tuple(row_cells))
                return tuple(result)
            mock_c = _mock_cell(cells.get(key))
            return mock_c
        ws.__getitem__ = getitem
    elif range_data:
        def getitem(_self, key):
            if key in range_data:
                rows = range_data[key]
                result = []
                for row_vals in rows:
                    row_cells = []
                    for val in row_vals:
                        row_cells.append(_mock_cell(val))
                    result.append(tuple(row_cells))
                return tuple(result)
            return _mock_cell(None)
        ws.__getitem__ = getitem
    else:
        ws.__getitem__ = lambda _self, key: _mock_cell(None)

    ws.merged_cells = MagicMock()
    ws.merged_cells.ranges = merged_ranges or []

    return ws


def _mock_workbook(
    sheets: dict[str, MagicMock] | None = None,
    active_sheet: MagicMock | None = None,
) -> MagicMock:
    """Create a mock openpyxl workbook."""
    wb = MagicMock()
    sheets = sheets or {}

    def getitem(_self, key):
        if key in sheets:
            return sheets[key]
        raise KeyError(key)

    wb.__getitem__ = getitem
    wb.active = active_sheet or (next(iter(sheets.values())) if sheets else MagicMock())
    return wb


# ---------------------------------------------------------------------------
# Excel Cell Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_excel_extract_text_field_single_cell(form_config):
    """Single cell 'B2' with text -> ExtractedField with value, confidence=0.95."""
    ws = _mock_worksheet(cells={"B2": "John Doe"})
    wb = _mock_workbook(sheets={"Sheet1": ws}, active_sheet=ws)

    fld = _excel_field(cell="B2")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert len(results) == 1
    assert results[0].value == "John Doe"
    assert results[0].confidence == 0.95
    assert results[0].extraction_method == "cell_mapping"
    assert results[0].field_name == "employee_name"


@pytest.mark.unit
def test_excel_extract_number_field_coercion(form_config):
    """Cell contains numeric value -> coerced to float."""
    ws = _mock_worksheet(cells={"C3": 42.5})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="salary", field_type=FieldType.NUMBER, cell="C3")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == 42.5
    assert isinstance(results[0].value, float)
    assert results[0].confidence == 0.95


@pytest.mark.unit
def test_excel_extract_number_field_coercion_failure(form_config):
    """Cell contains 'abc' for NUMBER -> value=None, W_FORM_FIELD_TYPE_COERCION."""
    ws = _mock_worksheet(cells={"C3": "abc"})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="amount", field_type=FieldType.NUMBER, cell="C3")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value is None
    assert FormErrorCode.W_FORM_FIELD_TYPE_COERCION.value in results[0].warnings


@pytest.mark.unit
def test_excel_extract_date_field(form_config):
    """Cell contains datetime object -> returned as ISO string."""
    dt = datetime(2026, 2, 15, 0, 0, 0)
    ws = _mock_worksheet(cells={"D4": dt})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="hire_date", field_type=FieldType.DATE, cell="D4")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "2026-02-15T00:00:00"
    assert results[0].confidence == 0.95


@pytest.mark.unit
@pytest.mark.parametrize("input_val", ["X", "x", "Yes", "TRUE", 1, True])
def test_excel_extract_checkbox_true_values(form_config, input_val):
    """Various truthy values -> True for CHECKBOX."""
    ws = _mock_worksheet(cells={"E5": input_val})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="agree", field_type=FieldType.CHECKBOX, cell="E5")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value is True


@pytest.mark.unit
@pytest.mark.parametrize("input_val", ["No", "FALSE", 0, False])
def test_excel_extract_checkbox_false_values(form_config, input_val):
    """Various falsy values -> False for CHECKBOX."""
    ws = _mock_worksheet(cells={"E5": input_val})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="agree", field_type=FieldType.CHECKBOX, cell="E5")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value is False


@pytest.mark.unit
def test_excel_extract_cell_range(form_config):
    """Range 'D5:D7' with values -> joined with newline."""
    ws = _mock_worksheet(
        range_data={"D5:D7": [["Line 1"], ["Line 2"], ["Line 3"]]},
    )
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="notes", cell="D5:D7")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "Line 1\nLine 2\nLine 3"
    assert results[0].confidence == 0.95


@pytest.mark.unit
def test_excel_extract_cell_range_skips_empty(form_config):
    """Range with some empty cells -> only non-empty joined."""
    ws = _mock_worksheet(
        range_data={"D5:D7": [["First"], [None], ["Third"]]},
    )
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="notes", cell="D5:D7")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "First\nThird"


@pytest.mark.unit
def test_excel_merged_cell_resolution(form_config):
    """Cell in merged range -> reads top-left, emits W_FORM_MERGED_CELL_RESOLVED."""
    ws = MagicMock()

    merged_range = MagicMock()
    merged_range.__contains__ = lambda self, item: item == "B3"
    merged_range.min_row = 2
    merged_range.min_col = 2

    ws.merged_cells = MagicMock()
    ws.merged_cells.ranges = [merged_range]

    top_left_cell = _mock_cell("Merged Value")
    ws.cell.return_value = top_left_cell

    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="dept", cell="B3")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "Merged Value"
    assert FormErrorCode.W_FORM_MERGED_CELL_RESOLVED.value in results[0].warnings


@pytest.mark.unit
def test_excel_empty_required_field(form_config):
    """Empty cell + required=True -> confidence=0.0, W_FORM_FIELD_MISSING_REQUIRED."""
    ws = _mock_worksheet(cells={"B2": None})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(field_name="ssn", required=True, cell="B2")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value is None
    assert results[0].confidence == 0.0
    assert FormErrorCode.W_FORM_FIELD_MISSING_REQUIRED.value in results[0].warnings


@pytest.mark.unit
def test_excel_empty_optional_field(form_config):
    """Empty cell + required=False + default_value -> confidence=0.95, value=default."""
    ws = _mock_worksheet(cells={"B2": None})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(
        field_name="dept", required=False, default_value="General", cell="B2"
    )
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "General"
    assert results[0].confidence == 0.95


@pytest.mark.unit
def test_excel_validation_pattern_pass(form_config):
    """Value matches regex -> validation_passed=True."""
    ws = _mock_worksheet(cells={"B2": "123-45-6789"})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(
        field_name="ssn",
        cell="B2",
        validation_pattern=r"\d{3}-\d{2}-\d{4}",
    )
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].validation_passed is True
    assert results[0].value == "123-45-6789"


@pytest.mark.unit
def test_excel_validation_pattern_fail(form_config):
    """Value fails regex -> validation_passed=False, W_FORM_FIELD_VALIDATION_FAILED."""
    ws = _mock_worksheet(cells={"B2": "not-valid"})
    wb = _mock_workbook(active_sheet=ws)

    fld = _excel_field(
        field_name="ssn",
        cell="B2",
        validation_pattern=r"\d{3}-\d{2}-\d{4}",
    )
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].validation_passed is False
    assert FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value in results[0].warnings


@pytest.mark.unit
def test_excel_sheet_name_resolution(form_config):
    """Field with explicit sheet_name -> reads from that sheet."""
    ws_data = _mock_worksheet(cells={"A1": "From Data Sheet"})
    ws_other = _mock_worksheet(cells={"A1": "Wrong Sheet"})
    wb = _mock_workbook(
        sheets={"Data": ws_data, "Other": ws_other},
        active_sheet=ws_other,
    )

    fld = _excel_field(field_name="info", cell="A1", sheet_name="Data")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value == "From Data Sheet"


@pytest.mark.unit
def test_excel_sheet_name_not_found(form_config):
    """Invalid sheet_name -> value=None, confidence=0.0, warning."""
    ws = _mock_worksheet(cells={"A1": "Some Value"})
    wb = _mock_workbook(sheets={"Sheet1": ws}, active_sheet=ws)

    fld = _excel_field(field_name="info", cell="A1", sheet_name="NonExistent")
    template = _excel_template(fields=[fld])
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert results[0].value is None
    assert results[0].confidence == 0.0
    assert any(
        FormErrorCode.E_FORM_EXTRACTION_FAILED.value in w for w in results[0].warnings
    )


@pytest.mark.unit
def test_excel_multiple_fields_extraction(form_config):
    """Template with 3 fields -> all extracted in order."""
    ws = _mock_worksheet(cells={"B2": "John", "C3": "Engineering", "D4": "42"})
    wb = _mock_workbook(active_sheet=ws)

    fields = [
        _excel_field(field_name="name", cell="B2"),
        _excel_field(field_name="dept", cell="C3"),
        _excel_field(field_name="age", field_type=FieldType.NUMBER, cell="D4"),
    ]
    template = _excel_template(fields=fields)
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert len(results) == 3
    assert results[0].field_name == "name"
    assert results[0].value == "John"
    assert results[1].field_name == "dept"
    assert results[1].value == "Engineering"
    assert results[2].field_name == "age"
    assert results[2].value == 42.0


@pytest.mark.unit
def test_excel_skips_pdf_fields(form_config):
    """Template mixing cell_address and region fields -> only cell_address processed."""
    ws = _mock_worksheet(cells={"B2": "Excel Value"})
    wb = _mock_workbook(active_sheet=ws)

    excel_fld = _excel_field(field_name="excel_field", cell="B2")
    pdf_fld = _field(field_name="pdf_field")  # region-based, no cell_address

    template = FormTemplate(
        name="Mixed Template",
        source_format=SourceFormat.XLSX,
        page_count=1,
        fields=[excel_fld, pdf_fld],
    )
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.return_value = wb
        results = extractor.extract("/fake/form.xlsx", template)

    assert len(results) == 1
    assert results[0].field_name == "excel_field"
    assert results[0].value == "Excel Value"


@pytest.mark.unit
def test_excel_corrupt_workbook_raises_exception(form_config):
    """Corrupt workbook -> FormIngestException with E_FORM_FILE_CORRUPT."""
    template = _excel_template()
    extractor = ExcelCellExtractor(config=form_config)

    with patch("ingestkit_forms.extractors.excel_cell.openpyxl") as mock_openpyxl:
        mock_openpyxl.load_workbook.side_effect = Exception("File is corrupt")
        with pytest.raises(FormIngestException) as exc_info:
            extractor.extract("/fake/corrupt.xlsx", template)

    assert exc_info.value.code == FormErrorCode.E_FORM_FILE_CORRUPT


@pytest.mark.unit
def test_excel_validation_redos_protection():
    """ReDoS pattern with pathological input -> timeout or no match."""
    result = regex_match_with_timeout(
        r"(a+)+$",
        "a" * 25 + "!",
        timeout=1.0,
    )
    # Either times out (None) or fails to match (False)
    assert result is None or result is False


# ===========================================================================
# VLM Fallback Extraction Tests (issue #66)
# ===========================================================================


from ingestkit_forms.extractors.vlm_fallback import (
    VLMFieldExtractor,
    _crop_field_region_with_padding,
)
from ingestkit_forms.protocols import VLMFieldResult

from tests.conftest import MockVLMBackend


# ---------------------------------------------------------------------------
# VLM test helpers
# ---------------------------------------------------------------------------


def _make_vlm_extracted_field(
    field_id: str = "f1",
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    value: str | bool | None = "OCR Value",
    confidence: float = 0.3,
    extraction_method: str = "ocr_overlay",
    page_number: int = 0,
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.3,
    height: float = 0.05,
    warnings: list[str] | None = None,
) -> ExtractedField:
    return ExtractedField(
        field_id=field_id,
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        value=value,
        confidence=confidence,
        extraction_method=extraction_method,
        bounding_box=BoundingBox(x=x, y=y, width=width, height=height),
        warnings=warnings if warnings is not None else [],
    )


def _make_vlm_template(field_mappings: list[FieldMapping]) -> FormTemplate:
    """Create a template with the given field mappings."""
    return FormTemplate(
        name="VLM Test Template",
        source_format=SourceFormat.PDF,
        page_count=1,
        fields=field_mappings,
    )


def _make_vlm_field_mapping(
    field_id: str = "f1",
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    page_number: int = 0,
    required: bool = False,
    extraction_hint: str | None = None,
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.3,
    height: float = 0.05,
) -> FieldMapping:
    return FieldMapping(
        field_id=field_id,
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        page_number=page_number,
        region=BoundingBox(x=x, y=y, width=width, height=height),
        required=required,
        extraction_hint=extraction_hint,
    )


# ---------------------------------------------------------------------------
# VLM Fallback Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVLMFallback:
    """Tests for VLMFieldExtractor.apply_vlm_fallback()."""

    def test_vlm_fallback_disabled(self, form_config):
        """When form_vlm_enabled=False, fields returned unchanged, no VLM calls."""
        backend = MockVLMBackend()
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=form_config)
        result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert len(result) == 1
        assert result[0].value == "OCR Value"
        assert result[0].confidence == 0.2
        assert backend.call_count == 0
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value not in result[0].warnings

    def test_vlm_fallback_no_low_confidence_fields(self, vlm_enabled_config):
        """All fields above threshold -> no VLM calls."""
        backend = MockVLMBackend()
        m1 = _make_vlm_field_mapping(field_id="f1", field_name="name")
        m2 = _make_vlm_field_mapping(field_id="f2", field_name="dept")
        template = _make_vlm_template([m1, m2])

        f1 = _make_vlm_extracted_field(field_id="f1", field_name="name", confidence=0.85)
        f2 = _make_vlm_extracted_field(field_id="f2", field_name="dept", confidence=0.92)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        result = extractor.apply_vlm_fallback([f1, f2], template, "/fake/form.pdf")

        assert len(result) == 2
        assert result[0].confidence == 0.85
        assert result[1].confidence == 0.92
        assert backend.call_count == 0

    def test_vlm_fallback_improves_field(self, vlm_enabled_config):
        """Low-confidence OCR field replaced when VLM returns high confidence."""
        backend = MockVLMBackend(default_value="VLM Value", default_confidence=0.85)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value == "VLM Value"
        assert result[0].confidence == 0.85
        assert result[0].extraction_method == "vlm_fallback"
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in result[0].warnings
        assert backend.call_count == 1

    def test_vlm_fallback_no_improvement(self, vlm_enabled_config):
        """VLM confidence below min threshold -> original OCR result retained."""
        backend = MockVLMBackend(default_value="Bad VLM", default_confidence=0.35)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value == "OCR Value"
        assert result[0].confidence == 0.2
        assert result[0].extraction_method == "ocr_overlay"
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in result[0].warnings

    def test_vlm_fallback_budget_exhausted(self, vlm_enabled_config):
        """More fields than budget -> overflow fields get W_FORM_VLM_BUDGET_EXHAUSTED."""
        vlm_enabled_config = FormProcessorConfig(
            form_vlm_enabled=True,
            form_vlm_fallback_threshold=0.4,
            form_vlm_max_fields_per_document=3,
            form_vlm_timeout_seconds=15,
        )
        backend = MockVLMBackend(default_value="VLM", default_confidence=0.85)

        mappings = []
        fields = []
        for i in range(5):
            fid = f"f{i}"
            m = _make_vlm_field_mapping(field_id=fid, field_name=f"field_{i}")
            mappings.append(m)
            f = _make_vlm_extracted_field(
                field_id=fid, field_name=f"field_{i}", confidence=0.2
            )
            fields.append(f)

        template = _make_vlm_template(mappings)
        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback(fields, template, "/fake/form.pdf")

        assert backend.call_count == 3

        # Count fields updated by VLM vs budget-exhausted
        vlm_updated = [r for r in result if r.extraction_method == "vlm_fallback"]
        budget_exhausted = [
            r
            for r in result
            if FormErrorCode.W_FORM_VLM_BUDGET_EXHAUSTED.value in r.warnings
        ]
        assert len(vlm_updated) == 3
        assert len(budget_exhausted) == 2

        # VLM-processed fields have the fallback warning
        for r in vlm_updated:
            assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in r.warnings

    def test_vlm_fallback_priority_required_first(self, vlm_enabled_config):
        """Required fields processed before optional, lowest confidence first."""
        vlm_enabled_config = FormProcessorConfig(
            form_vlm_enabled=True,
            form_vlm_fallback_threshold=0.4,
            form_vlm_max_fields_per_document=2,
            form_vlm_timeout_seconds=15,
        )
        backend = MockVLMBackend(default_value="VLM", default_confidence=0.85)

        # Field A: optional, conf 0.1
        ma = _make_vlm_field_mapping(field_id="fA", field_name="field_a", required=False)
        fa = _make_vlm_extracted_field(field_id="fA", field_name="field_a", confidence=0.1)

        # Field B: required, conf 0.3
        mb = _make_vlm_field_mapping(field_id="fB", field_name="field_b", required=True)
        fb = _make_vlm_extracted_field(field_id="fB", field_name="field_b", confidence=0.3)

        # Field C: required, conf 0.15
        mc = _make_vlm_field_mapping(field_id="fC", field_name="field_c", required=True)
        fc = _make_vlm_extracted_field(field_id="fC", field_name="field_c", confidence=0.15)

        # Field D: optional, conf 0.05
        md = _make_vlm_field_mapping(field_id="fD", field_name="field_d", required=False)
        fd = _make_vlm_extracted_field(field_id="fD", field_name="field_d", confidence=0.05)

        template = _make_vlm_template([ma, mb, mc, md])
        fields = [fa, fb, fc, fd]

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback(fields, template, "/fake/form.pdf")

        assert backend.call_count == 2
        # First call: field C (required, lowest conf 0.15)
        assert backend.call_args[0]["field_name"] == "field_c"
        # Second call: field B (required, conf 0.3)
        assert backend.call_args[1]["field_name"] == "field_b"

        # Fields A and D get budget exhausted
        result_a = result[0]  # field_a at index 0
        result_d = result[3]  # field_d at index 3
        assert FormErrorCode.W_FORM_VLM_BUDGET_EXHAUSTED.value in result_a.warnings
        assert FormErrorCode.W_FORM_VLM_BUDGET_EXHAUSTED.value in result_d.warnings

    def test_vlm_fallback_timeout_graceful(self, vlm_enabled_config, caplog):
        """On VLM TimeoutError, original OCR result retained."""
        backend = MockVLMBackend(raise_timeout=True)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with (
            patch(
                "ingestkit_forms.extractors.vlm_fallback.get_page_image",
                return_value=img,
            ),
            caplog.at_level(logging.WARNING, logger="ingestkit_forms"),
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value == "OCR Value"
        assert result[0].confidence == 0.2
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in result[0].warnings
        assert FormErrorCode.E_FORM_VLM_TIMEOUT.value in caplog.text

    def test_vlm_fallback_error_graceful(self, vlm_enabled_config):
        """On VLM generic exception, original OCR result retained."""
        backend = MockVLMBackend(raise_error=True)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value == "OCR Value"
        assert result[0].confidence == 0.2
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in result[0].warnings

    def test_vlm_fallback_unavailable(self, vlm_enabled_config, caplog):
        """When is_available() returns False, all fields returned unchanged."""
        backend = MockVLMBackend(available=False)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(field_id="f1", confidence=0.2)

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)

        with caplog.at_level(logging.WARNING, logger="ingestkit_forms"):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value == "OCR Value"
        assert result[0].confidence == 0.2
        assert backend.call_count == 0
        assert "VLM backend unavailable" in caplog.text

    def test_vlm_fallback_checkbox_field(self, vlm_enabled_config):
        """Checkbox field with low confidence correctly handled by VLM returning bool."""
        backend = MockVLMBackend(default_value=True, default_confidence=0.90)
        mapping = _make_vlm_field_mapping(
            field_id="f1",
            field_name="agree",
            field_label="I Agree",
            field_type=FieldType.CHECKBOX,
        )
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(
            field_id="f1",
            field_name="agree",
            field_label="I Agree",
            field_type=FieldType.CHECKBOX,
            value=False,
            confidence=0.15,
        )

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert result[0].value is True
        assert result[0].confidence == 0.90
        assert result[0].extraction_method == "vlm_fallback"

    def test_vlm_fallback_warnings_appended(self, vlm_enabled_config):
        """W_FORM_VLM_FALLBACK_USED appended to existing warnings, not replacing."""
        backend = MockVLMBackend(default_value="VLM", default_confidence=0.85)
        mapping = _make_vlm_field_mapping(field_id="f1")
        template = _make_vlm_template([mapping])
        field = _make_vlm_extracted_field(
            field_id="f1",
            confidence=0.2,
            warnings=[FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value],
        )

        extractor = VLMFieldExtractor(vlm_backend=backend, config=vlm_enabled_config)
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))

        with patch(
            "ingestkit_forms.extractors.vlm_fallback.get_page_image",
            return_value=img,
        ):
            result = extractor.apply_vlm_fallback([field], template, "/fake/form.pdf")

        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in result[0].warnings
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in result[0].warnings

    def test_crop_with_padding(self):
        """Verify 10% padding applied correctly to crop coordinates."""
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        region = BoundingBox(x=0.2, y=0.3, width=0.4, height=0.2)
        # Pixel: x=200, y=300, w=400, h=200
        # Padding: pad_x=40, pad_y=20
        # Crop: (160, 280, 640, 520) -> size (480, 240)
        cropped = _crop_field_region_with_padding(img, region)
        assert cropped.size == (480, 240)

    def test_crop_with_padding_edge(self):
        """Padding at image edge clamped to bounds."""
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        region = BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)
        # Pixel: x=0, y=0, w=100, h=100
        # Padding: pad_x=10, pad_y=10
        # Crop: (max(0,-10)=0, max(0,-10)=0, min(1000,110)=110, min(1000,110)=110)
        cropped = _crop_field_region_with_padding(img, region)
        assert cropped.size == (110, 110)
