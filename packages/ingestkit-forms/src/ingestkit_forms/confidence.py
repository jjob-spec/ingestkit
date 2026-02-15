"""Per-field confidence scoring and overall confidence aggregation.

Implements the confidence rules from spec section 7.4:
- Per-field confidence adjustment based on extraction method
- 4-tier confidence-based action system
- Weighted overall confidence aggregation (required fields get 2x weight)

See also: ``config.py`` for threshold parameters, ``errors.py`` for warning codes.
"""

from __future__ import annotations

import logging

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode
from ingestkit_forms.models import ExtractedField, FieldType, FormTemplate

logger = logging.getLogger("ingestkit_forms")


def compute_field_confidence(
    extraction_method: str,
    raw_confidence: float,
    field_type: FieldType,
    coercion_applied: bool = False,
) -> float:
    """Clamp/adjust raw confidence to spec-defined ranges per extraction method.

    Spec section 7.4 confidence ranges:
        - native_fields: 0.90-0.99, deduct 0.02 if coercion applied
        - ocr_overlay: pass through raw OCR confidence (already correct)
        - cell_mapping: 0.90-0.99, deduct 0.02 if coercion applied
        - vlm_fallback: pass through raw VLM confidence
        - checkbox/radio/signature via OCR: pass through (already computed)

    Args:
        extraction_method: One of 'native_fields', 'ocr_overlay',
            'cell_mapping', 'vlm_fallback'.
        raw_confidence: The raw confidence value from the extractor.
        field_type: The field's data type.
        coercion_applied: Whether type coercion was performed on the value.

    Returns:
        Adjusted confidence value clamped to the appropriate range.
    """
    if extraction_method in ("native_fields", "cell_mapping"):
        confidence = max(0.90, min(raw_confidence, 0.99))
        if coercion_applied:
            confidence = max(0.90, confidence - 0.02)
        return confidence

    if extraction_method in ("ocr_overlay", "vlm_fallback"):
        # Pass through: OCR char-averaging and VLM confidence are already
        # computed correctly by the respective extractors.
        return max(0.0, min(raw_confidence, 1.0))

    # Unknown extraction method: pass through with basic clamping
    return max(0.0, min(raw_confidence, 1.0))


def compute_overall_confidence(
    fields: list[ExtractedField],
    template: FormTemplate,
) -> float:
    """Compute weighted mean confidence across all extracted fields.

    Required fields receive 2x weight; optional fields receive 1x weight.
    Implements the spec section 7.4 pseudocode.

    Args:
        fields: List of extracted field results.
        template: The form template (used to look up required flag).

    Returns:
        Weighted mean confidence in [0.0, 1.0]. Returns 0.0 for empty fields.
    """
    if not fields:
        return 0.0

    # Build field_id -> FieldMapping lookup from template
    field_map = {fm.field_id: fm for fm in template.fields}

    weighted_sum = 0.0
    total_weight = 0.0

    for ef in fields:
        # PLAN-CHECK correction: cannot instantiate bare FieldMapping() as default
        # because it requires region or cell_address. Instead, default to weight=1.0.
        weight = (
            2.0
            if (fm := field_map.get(ef.field_id)) and fm.required
            else 1.0
        )
        weighted_sum += ef.confidence * weight
        total_weight += weight

    return weighted_sum / max(total_weight, 1.0)


def apply_confidence_actions(
    field: ExtractedField,
    config: FormProcessorConfig,
) -> tuple[ExtractedField, list[str]]:
    """Apply the 4-tier confidence-based action rules from spec section 7.4.

    Tiers:
        1. confidence >= min_field_confidence -> accept as-is
        2. confidence >= vlm_fallback_threshold (but < min_field) -> accept + warning
        3. confidence < vlm_fallback_threshold AND vlm_enabled -> mark for VLM fallback
        4. confidence < vlm_fallback_threshold AND vlm disabled -> value=None + warning

    Args:
        field: The extracted field to evaluate.
        config: Processor config with threshold values.

    Returns:
        Tuple of (possibly modified field, list of warning code strings).
        The field is returned as a new instance if modifications are needed.
    """
    warnings: list[str] = []
    confidence = field.confidence

    # Tier 1: Accept as-is
    if confidence >= config.form_extraction_min_field_confidence:
        return field, warnings

    # Tier 2: Accept with warning (between vlm_threshold and min_field)
    if confidence >= config.form_vlm_fallback_threshold:
        warnings.append(FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value)
        return field, warnings

    # Tier 3: Below vlm_threshold, VLM enabled -> mark for VLM fallback
    if config.form_vlm_enabled:
        warnings.append(FormErrorCode.W_FORM_VLM_FALLBACK_USED.value)
        # Return field with extraction_method updated to signal VLM is needed.
        # The router will handle actual VLM invocation.
        updated = field.model_copy(
            update={"extraction_method": "vlm_fallback_pending"}
        )
        return updated, warnings

    # Tier 4: Below vlm_threshold, VLM disabled -> value=None + warning
    warnings.append(FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value)
    updated = field.model_copy(update={"value": None})
    return updated, warnings
