"""VLM fallback extractor for low-confidence OCR fields.

Post-processes OCR extraction results by re-extracting fields with
confidence below form_vlm_fallback_threshold using a Vision-Language
Model backend. See spec section 7.5.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from ingestkit_forms.errors import FormErrorCode
from ingestkit_forms.extractors._rendering import get_page_image
from ingestkit_forms.models import BoundingBox, ExtractedField

if TYPE_CHECKING:
    from PIL import Image

    from ingestkit_forms.config import FormProcessorConfig
    from ingestkit_forms.models import FieldMapping, FormTemplate
    from ingestkit_forms.protocols import VLMBackend

logger = logging.getLogger("ingestkit_forms")


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _crop_field_region_with_padding(
    page_image: Image.Image,
    region: BoundingBox,
    padding_pct: float = 0.10,
) -> Image.Image:
    """Crop field region with padding for VLM context (spec section 7.5 step 1).

    Expands the bounding box by padding_pct on each side, clamped to
    image dimensions. Default 10% padding provides surrounding context
    for the VLM to interpret the field.

    Args:
        page_image: Full page image (PIL Image).
        region: Normalized bounding box (0.0-1.0 coordinates).
        padding_pct: Fractional padding to add on each side (default 0.10).

    Returns:
        Cropped PIL Image with padding.
    """
    page_w, page_h = page_image.size

    # Convert normalized coords to pixels
    px_x = region.x * page_w
    px_y = region.y * page_h
    px_w = region.width * page_w
    px_h = region.height * page_h

    # Compute padding in pixels (relative to field dimensions)
    pad_x = px_w * padding_pct
    pad_y = px_h * padding_pct

    # Expand with padding, clamp to image bounds
    left = max(0, int(px_x - pad_x))
    top = max(0, int(px_y - pad_y))
    right = min(page_w, int(px_x + px_w + pad_x))
    bottom = min(page_h, int(px_y + px_h + pad_y))

    return page_image.crop((left, top, right, bottom))


def _image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to PNG-encoded bytes for VLM backend."""
    buf = io.BytesIO()
    if image.mode == "1":
        image = image.convert("L")
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# VLMFieldExtractor
# ---------------------------------------------------------------------------


class VLMFieldExtractor:
    """Post-processes OCR results using VLM for low-confidence fields.

    Standalone class called after OCR extraction. Iterates the extraction
    results, identifies fields below the VLM threshold, and re-extracts
    them using the VLMBackend protocol with padded field crops.

    All VLM access is via the VLMBackend protocol (never concrete imports).
    See spec section 7.5.
    """

    def __init__(
        self,
        vlm_backend: VLMBackend,
        config: FormProcessorConfig,
    ) -> None:
        self._vlm = vlm_backend
        self._config = config

    def apply_vlm_fallback(
        self,
        fields: list[ExtractedField],
        template: FormTemplate,
        file_path: str,
    ) -> list[ExtractedField]:
        """Apply VLM fallback extraction to low-confidence OCR fields.

        Algorithm (spec section 7.5):
        1. Guard: if not form_vlm_enabled, return unchanged.
        2. Availability check: if vlm_backend.is_available() is False, log
           and return unchanged.
        3. Identify candidates: fields with confidence < form_vlm_fallback_threshold.
        4. Priority sort: required first, lowest confidence first.
        5. Budget: take first form_vlm_max_fields_per_document candidates.
           Emit W_FORM_VLM_BUDGET_EXHAUSTED for overflow.
        6. For each candidate: crop, call VLM, replace if improved.
        7. Return updated field list.

        Args:
            fields: List of ExtractedField from OCR extraction.
            template: FormTemplate with field mappings.
            file_path: Path to the source document (PDF or image).

        Returns:
            Updated list of ExtractedField with VLM results where improved.
        """
        # Step 1 -- Guard clause
        if not self._config.form_vlm_enabled:
            return fields

        # Step 2 -- Availability check
        if not self._vlm.is_available():
            logger.warning(
                "VLM backend unavailable, skipping fallback for %d fields",
                len(fields),
            )
            return fields

        # Step 3 -- Identify candidates
        threshold = self._config.form_vlm_fallback_threshold
        # Build field mapping lookup
        mapping_by_id: dict[str, FieldMapping] = {
            m.field_id: m for m in template.fields
        }

        # (index, field) pairs for candidates
        candidates: list[tuple[int, ExtractedField]] = []
        for idx, field in enumerate(fields):
            if field.confidence < threshold:
                candidates.append((idx, field))

        if not candidates:
            return fields

        # Step 4 -- Priority sort: required first, lowest confidence first
        def _sort_key(item: tuple[int, ExtractedField]) -> tuple[bool, float]:
            _idx, fld = item
            mapping = mapping_by_id.get(fld.field_id)
            is_required = mapping.required if mapping else False
            return (not is_required, fld.confidence)

        candidates.sort(key=_sort_key)

        # Step 5 -- Budget enforcement
        budget = self._config.form_vlm_max_fields_per_document
        within_budget = candidates[:budget]
        over_budget = candidates[budget:]

        # Build output list as a copy
        result = list(fields)

        # Mark over-budget fields
        for idx, _field in over_budget:
            original = result[idx]
            updated_warnings = list(original.warnings)
            updated_warnings.append(FormErrorCode.W_FORM_VLM_BUDGET_EXHAUSTED.value)
            result[idx] = original.model_copy(update={"warnings": updated_warnings})

        # Step 6 -- Process each candidate within budget
        page_cache: dict[int, Image.Image] = {}

        for idx, original_field in within_budget:
            mapping = self._get_field_mapping(original_field.field_id, template)
            if mapping is None:
                logger.warning(
                    "No field mapping found for field_id=%s, skipping VLM fallback",
                    original_field.field_id,
                )
                continue

            page_number = mapping.page_number

            # Render or cache page image
            if page_number not in page_cache:
                try:
                    page_cache[page_number] = get_page_image(
                        file_path, page_number, dpi=self._config.form_ocr_dpi
                    )
                except Exception:
                    logger.warning(
                        "Failed to render page %d for VLM fallback, "
                        "retaining OCR results for all fields on this page",
                        page_number,
                    )
                    # Append VLM fallback used warning even on render failure
                    updated_warnings = list(original_field.warnings)
                    updated_warnings.append(
                        FormErrorCode.W_FORM_VLM_FALLBACK_USED.value
                    )
                    result[idx] = original_field.model_copy(
                        update={"warnings": updated_warnings}
                    )
                    continue

            page_image = page_cache[page_number]

            if mapping.region is None:
                logger.warning(
                    "Field '%s' has no bounding box region, skipping VLM fallback",
                    mapping.field_name,
                )
                continue

            # Crop with 10% padding
            cropped = _crop_field_region_with_padding(page_image, mapping.region)
            image_bytes = _image_to_png_bytes(cropped)

            # Call VLM backend
            updated_warnings = list(original_field.warnings)
            updated_warnings.append(FormErrorCode.W_FORM_VLM_FALLBACK_USED.value)

            try:
                vlm_result = self._vlm.extract_field(
                    image_bytes=image_bytes,
                    field_type=mapping.field_type.value,
                    field_name=mapping.field_name,
                    extraction_hint=mapping.extraction_hint,
                    timeout=float(self._config.form_vlm_timeout_seconds),
                )

                if vlm_result.confidence >= self._config.form_extraction_min_field_confidence:
                    # VLM improved the result -- replace
                    if self._config.log_sample_data:
                        logger.debug(
                            "VLM improved field '%s': confidence %.2f -> %.2f, "
                            "value='%s' -> '%s'",
                            original_field.field_name,
                            original_field.confidence,
                            vlm_result.confidence,
                            original_field.value,
                            vlm_result.value,
                        )
                    else:
                        logger.debug(
                            "VLM improved field '%s': confidence %.2f -> %.2f",
                            original_field.field_name,
                            original_field.confidence,
                            vlm_result.confidence,
                        )

                    result[idx] = ExtractedField(
                        field_id=original_field.field_id,
                        field_name=original_field.field_name,
                        field_label=original_field.field_label,
                        field_type=original_field.field_type,
                        value=vlm_result.value,
                        raw_value=(
                            str(vlm_result.value)
                            if vlm_result.value is not None
                            else None
                        ),
                        confidence=vlm_result.confidence,
                        extraction_method="vlm_fallback",
                        bounding_box=original_field.bounding_box,
                        validation_passed=None,
                        warnings=updated_warnings,
                    )
                else:
                    # VLM did not improve -- retain original with warning
                    if self._config.log_sample_data:
                        logger.debug(
                            "VLM did not improve field '%s': "
                            "VLM confidence %.2f < min %.2f, value='%s'",
                            original_field.field_name,
                            vlm_result.confidence,
                            self._config.form_extraction_min_field_confidence,
                            vlm_result.value,
                        )
                    else:
                        logger.debug(
                            "VLM did not improve field '%s': "
                            "VLM confidence %.2f < min %.2f",
                            original_field.field_name,
                            vlm_result.confidence,
                            self._config.form_extraction_min_field_confidence,
                        )
                    result[idx] = original_field.model_copy(
                        update={"warnings": updated_warnings}
                    )

            except TimeoutError:
                logger.warning(
                    "%s: VLM timeout for field '%s' (timeout=%ds)",
                    FormErrorCode.E_FORM_VLM_TIMEOUT.value,
                    original_field.field_name,
                    self._config.form_vlm_timeout_seconds,
                )
                result[idx] = original_field.model_copy(
                    update={"warnings": updated_warnings}
                )

            except Exception:
                logger.warning(
                    "%s: VLM error for field '%s'",
                    FormErrorCode.E_FORM_VLM_UNAVAILABLE.value,
                    original_field.field_name,
                    exc_info=True,
                )
                result[idx] = original_field.model_copy(
                    update={"warnings": updated_warnings}
                )

        return result

    def _get_field_mapping(
        self,
        field_id: str,
        template: FormTemplate,
    ) -> FieldMapping | None:
        """Look up a FieldMapping from the template by field_id."""
        for f in template.fields:
            if f.field_id == field_id:
                return f
        return None
