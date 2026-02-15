"""Tests for per-field confidence scoring and overall confidence aggregation.

Tests the three public functions in confidence.py:
- compute_field_confidence: per-method confidence adjustment
- compute_overall_confidence: weighted mean with 2x required fields
- apply_confidence_actions: 4-tier confidence action system
"""

from __future__ import annotations

import pytest

from ingestkit_forms.confidence import (
    apply_confidence_actions,
    compute_field_confidence,
    compute_overall_confidence,
)
from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode
from ingestkit_forms.models import (
    BoundingBox,
    ExtractedField,
    FieldMapping,
    FieldType,
    FormTemplate,
    SourceFormat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extracted_field(
    field_id: str = "f1",
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    value: str | bool | float | None = "John Doe",
    confidence: float = 0.95,
    extraction_method: str = "native_fields",
) -> ExtractedField:
    return ExtractedField(
        field_id=field_id,
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        value=value,
        confidence=confidence,
        extraction_method=extraction_method,
    )


def _field_mapping(
    field_id: str = "f1",
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    required: bool = False,
) -> FieldMapping:
    return FieldMapping(
        field_id=field_id,
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        page_number=0,
        region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
        required=required,
    )


def _template(fields: list[FieldMapping]) -> FormTemplate:
    return FormTemplate(
        name="Test Template",
        source_format=SourceFormat.PDF,
        page_count=1,
        fields=fields,
    )


# ===========================================================================
# compute_field_confidence tests
# ===========================================================================


@pytest.mark.unit
class TestComputeFieldConfidence:
    def test_native_pdf_base_confidence(self):
        """Native PDF: raw 0.99 -> clamped to 0.99."""
        result = compute_field_confidence("native_fields", 0.99, FieldType.TEXT)
        assert result == 0.99

    def test_native_pdf_clamp_to_range(self):
        """Native PDF: raw 1.0 -> clamped to 0.99."""
        result = compute_field_confidence("native_fields", 1.0, FieldType.TEXT)
        assert result == 0.99

    def test_native_pdf_clamp_low(self):
        """Native PDF: raw 0.80 -> clamped to 0.90 (floor)."""
        result = compute_field_confidence("native_fields", 0.80, FieldType.TEXT)
        assert result == 0.90

    def test_native_pdf_with_coercion(self):
        """Native PDF with coercion: 0.99 - 0.02 = 0.97."""
        result = compute_field_confidence(
            "native_fields", 0.99, FieldType.NUMBER, coercion_applied=True
        )
        assert result == pytest.approx(0.97)

    def test_native_pdf_coercion_respects_floor(self):
        """Native PDF with coercion: 0.91 - 0.02 = 0.90 (clamped to floor)."""
        result = compute_field_confidence(
            "native_fields", 0.91, FieldType.NUMBER, coercion_applied=True
        )
        assert result == pytest.approx(0.90)

    def test_ocr_pass_through(self):
        """OCR overlay: raw confidence passed through unchanged."""
        result = compute_field_confidence("ocr_overlay", 0.73, FieldType.TEXT)
        assert result == pytest.approx(0.73)

    def test_ocr_pass_through_low(self):
        """OCR overlay: very low raw confidence is passed through."""
        result = compute_field_confidence("ocr_overlay", 0.15, FieldType.TEXT)
        assert result == pytest.approx(0.15)

    def test_cell_mapping_base(self):
        """Cell mapping: raw 0.99 clamped to 0.90-0.99."""
        result = compute_field_confidence("cell_mapping", 0.99, FieldType.TEXT)
        assert result == 0.99

    def test_cell_mapping_with_coercion(self):
        """Cell mapping with coercion: deduction applied."""
        result = compute_field_confidence(
            "cell_mapping", 0.95, FieldType.NUMBER, coercion_applied=True
        )
        assert result == pytest.approx(0.93)

    def test_vlm_fallback_pass_through(self):
        """VLM fallback: raw confidence passed through."""
        result = compute_field_confidence("vlm_fallback", 0.65, FieldType.TEXT)
        assert result == pytest.approx(0.65)

    def test_checkbox_ocr_pass_through(self):
        """Checkbox via OCR: pass through (already computed by fill ratio)."""
        result = compute_field_confidence("ocr_overlay", 0.85, FieldType.CHECKBOX)
        assert result == pytest.approx(0.85)

    def test_unknown_method_pass_through(self):
        """Unknown extraction method: clamp to [0.0, 1.0], pass through."""
        result = compute_field_confidence("some_future_method", 0.7, FieldType.TEXT)
        assert result == pytest.approx(0.7)

    def test_negative_raw_clamped_to_zero(self):
        """Negative raw confidence clamped to 0.0 for OCR."""
        result = compute_field_confidence("ocr_overlay", -0.5, FieldType.TEXT)
        assert result == 0.0


# ===========================================================================
# compute_overall_confidence tests
# ===========================================================================


@pytest.mark.unit
class TestComputeOverallConfidence:
    def test_equal_confidence_all_optional(self):
        """All fields equal confidence (0.8), all optional -> returns 0.8."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.8),
            _extracted_field(field_id="f2", field_name="dept", confidence=0.8),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=False),
            _field_mapping(field_id="f2", field_name="dept", required=False),
        ]
        template = _template(mappings)
        result = compute_overall_confidence(fields, template)
        assert result == pytest.approx(0.8)

    def test_required_fields_get_double_weight(self):
        """2 required at 0.8, 1 optional at 0.4 -> (0.8*2 + 0.8*2 + 0.4*1) / 5 = 0.72."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.8),
            _extracted_field(field_id="f2", field_name="dept", confidence=0.8),
            _extracted_field(field_id="f3", field_name="notes", confidence=0.4),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=True),
            _field_mapping(field_id="f2", field_name="dept", required=True),
            _field_mapping(field_id="f3", field_name="notes", required=False),
        ]
        template = _template(mappings)
        result = compute_overall_confidence(fields, template)
        assert result == pytest.approx(0.72)

    def test_empty_fields_returns_zero(self):
        """No fields -> returns 0.0 (no division by zero)."""
        mappings = [_field_mapping()]
        template = _template(mappings)
        result = compute_overall_confidence([], template)
        assert result == 0.0

    def test_fields_with_none_value_and_low_confidence(self):
        """Fields with value=None and confidence=0.0 pull down the average."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.9),
            _extracted_field(field_id="f2", field_name="bad", value=None, confidence=0.0),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=False),
            _field_mapping(field_id="f2", field_name="bad", required=False),
        ]
        template = _template(mappings)
        result = compute_overall_confidence(fields, template)
        assert result == pytest.approx(0.45)

    def test_all_required_fields(self):
        """All fields required -> all get weight 2.0."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.9),
            _extracted_field(field_id="f2", field_name="dept", confidence=0.7),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=True),
            _field_mapping(field_id="f2", field_name="dept", required=True),
        ]
        template = _template(mappings)
        result = compute_overall_confidence(fields, template)
        # (0.9*2 + 0.7*2) / (2+2) = 3.2 / 4 = 0.8
        assert result == pytest.approx(0.8)

    def test_field_not_in_template_gets_weight_one(self):
        """Field with field_id not in template defaults to weight 1.0."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.9),
            _extracted_field(field_id="unknown_id", field_name="extra", confidence=0.5),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=True),
        ]
        template = _template(mappings)
        result = compute_overall_confidence(fields, template)
        # (0.9*2 + 0.5*1) / (2+1) = 2.3 / 3 ≈ 0.7667
        assert result == pytest.approx(2.3 / 3.0)

    def test_overall_below_threshold_detected(self):
        """Verify the function can produce values below the 0.3 threshold."""
        fields = [
            _extracted_field(field_id="f1", confidence=0.1),
            _extracted_field(field_id="f2", field_name="dept", confidence=0.2),
        ]
        mappings = [
            _field_mapping(field_id="f1", required=False),
            _field_mapping(field_id="f2", field_name="dept", required=False),
        ]
        template = _template(mappings)
        config = FormProcessorConfig()
        result = compute_overall_confidence(fields, template)
        assert result < config.form_extraction_min_overall_confidence


# ===========================================================================
# apply_confidence_actions tests
# ===========================================================================


@pytest.mark.unit
class TestApplyConfidenceActions:
    def test_tier1_accept_high_confidence(self):
        """Confidence 0.8 >= min_field 0.5 -> accept, no warnings."""
        config = FormProcessorConfig()  # min_field=0.5
        field = _extracted_field(confidence=0.8)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value == "John Doe"
        assert warnings == []

    def test_tier1_exact_boundary(self):
        """Confidence == min_field_confidence -> accept (inclusive)."""
        config = FormProcessorConfig()  # min_field=0.5
        field = _extracted_field(confidence=0.5)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value == "John Doe"
        assert warnings == []

    def test_tier2_accept_with_warning(self):
        """Confidence 0.45 >= vlm_threshold 0.4, < min_field 0.5 -> warning."""
        config = FormProcessorConfig()  # vlm_threshold=0.4, min_field=0.5
        field = _extracted_field(confidence=0.45)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value == "John Doe"  # Value preserved
        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in warnings

    def test_tier2_exact_boundary(self):
        """Confidence == vlm_fallback_threshold -> tier 2 (inclusive)."""
        config = FormProcessorConfig()  # vlm_threshold=0.4
        field = _extracted_field(confidence=0.4)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value == "John Doe"
        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in warnings

    def test_tier3_vlm_enabled_marks_for_fallback(self):
        """Confidence 0.3 < vlm_threshold 0.4, vlm_enabled -> mark for VLM."""
        config = FormProcessorConfig(form_vlm_enabled=True)
        field = _extracted_field(confidence=0.3)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.extraction_method == "vlm_fallback_pending"
        assert FormErrorCode.W_FORM_VLM_FALLBACK_USED.value in warnings
        # Value is preserved (VLM will replace it later)
        assert updated.value == "John Doe"

    def test_tier4_vlm_disabled_nulls_value(self):
        """Confidence 0.3 < vlm_threshold 0.4, vlm disabled -> value=None."""
        config = FormProcessorConfig()  # vlm_enabled=False by default
        field = _extracted_field(confidence=0.3)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value is None
        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in warnings

    def test_tier4_zero_confidence(self):
        """Confidence 0.0 -> tier 4 (value=None)."""
        config = FormProcessorConfig()
        field = _extracted_field(confidence=0.0)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value is None
        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in warnings

    def test_tier4_preserves_other_fields(self):
        """Tier 4 only modifies value; other fields remain intact."""
        config = FormProcessorConfig()
        field = _extracted_field(
            field_id="special",
            field_name="ssn",
            confidence=0.2,
            extraction_method="ocr_overlay",
        )
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.field_id == "special"
        assert updated.field_name == "ssn"
        assert updated.extraction_method == "ocr_overlay"
        assert updated.confidence == 0.2  # Confidence not modified
        assert updated.value is None

    def test_validation_failure_confidence_zero(self):
        """Field with confidence=0.0 from validation failure -> tier 4."""
        config = FormProcessorConfig()
        field = _extracted_field(confidence=0.0, value=None)
        updated, warnings = apply_confidence_actions(field, config)
        assert updated.value is None
        assert FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value in warnings
