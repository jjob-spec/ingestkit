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
    _regex_match_with_timeout,
)
from ingestkit_forms.models import (
    BoundingBox,
    CellAddress,
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
    assert results[0].confidence == 0.95
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
    assert results[0].confidence == 0.95


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
        result = _regex_match_with_timeout(
            r"(a+)+$",
            "a" * 25 + "!",
            timeout=1.0,
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
