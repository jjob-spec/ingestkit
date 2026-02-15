"""NativePDFExtractor: fillable PDF form field extraction.

Extracts values from fillable PDF form widgets using the PDFWidgetBackend
protocol. Supports text fields, checkboxes, radio buttons, dropdowns,
and signature fields (spec section 7.1).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode
from ingestkit_forms.models import (
    BoundingBox,
    ExtractedField,
    FieldMapping,
    FieldType,
    FormTemplate,
)
from ingestkit_forms.protocols import OCRBackend, PDFWidgetBackend, WidgetField

logger = logging.getLogger(__name__)


class NativePDFExtractor:
    """Extracts form field values from fillable PDFs using widget matching.

    Uses the injected PDFWidgetBackend to read form widgets, matches them
    to template FieldMapping regions via IoU scoring, and validates/coerces
    extracted values. Falls back to per-field OCR when available.
    """

    def __init__(
        self,
        pdf_backend: PDFWidgetBackend,
        config: FormProcessorConfig,
        ocr_backend: OCRBackend | None = None,
    ) -> None:
        self._pdf_backend = pdf_backend
        self._config = config
        self._ocr_backend = ocr_backend

    def extract(
        self,
        file_path: str,
        template: FormTemplate,
    ) -> list[ExtractedField]:
        """Extract form field values from a fillable PDF.

        Algorithm (spec section 7.1):
            1. Check has_form_fields(); if False -> return empty list
            2. Group template fields by page_number
            3. For each page: extract widgets, match to fields by IoU
            4. Validate and coerce values
            5. Return list[ExtractedField]
        """
        if not self._pdf_backend.has_form_fields(file_path):
            logger.warning("PDF has no form fields (flattened): %s", file_path)
            return []

        # Group fields by page, skipping Excel-only fields (no region)
        fields_by_page: dict[int, list[FieldMapping]] = defaultdict(list)
        for field in template.fields:
            if field.region is not None:
                fields_by_page[field.page_number].append(field)

        results: list[ExtractedField] = []
        for page_num, page_fields in sorted(fields_by_page.items()):
            widgets = self._pdf_backend.extract_widgets(file_path, page_num)
            for field_mapping in page_fields:
                extracted = self._match_and_extract(field_mapping, widgets, file_path)
                results.append(extracted)

        return results

    def _match_and_extract(
        self,
        field: FieldMapping,
        widgets: list[WidgetField],
        file_path: str,
    ) -> ExtractedField:
        """Match a single template field to the best-overlapping widget."""
        assert field.region is not None  # noqa: S101 – caller guarantees

        best_widget: WidgetField | None = None
        best_iou = 0.0

        for widget in widgets:
            iou = self._compute_iou(field.region, widget.bbox)
            if iou >= self._config.native_pdf_iou_threshold and iou > best_iou:
                best_iou = iou
                best_widget = widget

        if best_widget is not None:
            raw_value = best_widget.field_value
            coerced_value = self._coerce_widget_value(
                raw_value, field.field_type, best_widget.field_type
            )
            value, validation_passed, warnings = self._validate_field_value(
                coerced_value, field
            )
            return ExtractedField(
                field_id=field.field_id,
                field_name=field.field_name,
                field_label=field.field_label,
                field_type=field.field_type,
                value=value,
                raw_value=raw_value,
                confidence=0.95,
                extraction_method="native_fields",
                bounding_box=best_widget.bbox,
                validation_passed=validation_passed,
                warnings=warnings,
            )

        # No widget matched above IoU threshold
        return ExtractedField(
            field_id=field.field_id,
            field_name=field.field_name,
            field_label=field.field_label,
            field_type=field.field_type,
            value=None,
            raw_value=None,
            confidence=0.0,
            extraction_method=(
                "native_fields_with_ocr_fallback"
                if self._ocr_backend is not None
                else "native_fields"
            ),
            bounding_box=field.region,
            validation_passed=None,
            warnings=[FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value],
        )

    @staticmethod
    def _compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute Intersection over Union for two axis-aligned bounding boxes.

        Both boxes use normalized (0.0-1.0) coordinates with (x, y) as top-left.
        """
        # Convert (x, y, width, height) -> (x1, y1, x2, y2)
        a_x1, a_y1 = box_a.x, box_a.y
        a_x2, a_y2 = box_a.x + box_a.width, box_a.y + box_a.height

        b_x1, b_y1 = box_b.x, box_b.y
        b_x2, b_y2 = box_b.x + box_b.width, box_b.y + box_b.height

        # Intersection
        inter_x1 = max(a_x1, b_x1)
        inter_y1 = max(a_y1, b_y1)
        inter_x2 = min(a_x2, b_x2)
        inter_y2 = min(a_y2, b_y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Union
        area_a = box_a.width * box_a.height
        area_b = box_b.width * box_b.height
        union_area = area_a + area_b - inter_area

        if union_area == 0.0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _coerce_widget_value(
        raw_value: str | None,
        field_type: FieldType,
        widget_type: str,
    ) -> str | bool | float | None:
        """Coerce a raw PDF widget value to the expected FieldType.

        Args:
            raw_value: String value from the PDF widget (may be None for empty fields).
            field_type: Expected type from the template FieldMapping.
            widget_type: Widget's own type string ("text", "checkbox", "radio", etc.).
        """
        if raw_value is None:
            if field_type in (FieldType.CHECKBOX, FieldType.RADIO, FieldType.SIGNATURE):
                return False
            return None

        if field_type == FieldType.NUMBER:
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                return raw_value  # Let validation catch it

        if field_type in (FieldType.CHECKBOX, FieldType.RADIO):
            return raw_value.lower() in {"yes", "on", "true", "1"}

        if field_type == FieldType.SIGNATURE:
            return bool(raw_value.strip())

        # TEXT, DATE, DROPDOWN
        return raw_value

    @staticmethod
    def _validate_field_value(
        value: str | bool | float | None,
        field: FieldMapping,
    ) -> tuple[str | bool | float | None, bool | None, list[str]]:
        """Validate an extracted value against field constraints.

        Returns:
            (possibly_adjusted_value, validation_passed, warnings)
            - validation_passed: True if passed, False if failed, None if no pattern
            - If validation fails: value is kept (not nulled), warning is appended
        """
        if field.validation_pattern is None:
            return (value, None, [])

        if value is None:
            return (None, None, [])

        str_value = str(value)

        try:
            if re.fullmatch(field.validation_pattern, str_value):
                return (value, True, [])
            return (
                value,
                False,
                [FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value],
            )
        except re.error:
            return (
                value,
                None,
                [f"Invalid validation pattern: {field.validation_pattern}"],
            )
