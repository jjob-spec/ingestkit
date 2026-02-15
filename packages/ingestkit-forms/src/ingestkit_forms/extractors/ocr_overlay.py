"""OCROverlayExtractor: render + template overlay + per-field OCR.

Renders PDF pages or processes scanned images, overlays the template
field regions, and runs OCR on each field's bounding box.
See spec section 7.2.
"""

from __future__ import annotations

import io
import logging
import re
from typing import TYPE_CHECKING

from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.security import regex_match_with_timeout
from ingestkit_forms.extractors._preprocessing import (
    compute_fill_ratio,
    compute_ink_ratio,
    preprocess_for_ocr,
)
from ingestkit_forms.extractors._rendering import get_page_image
from ingestkit_forms.models import BoundingBox, ExtractedField, FieldType

if TYPE_CHECKING:
    from PIL import Image

    from ingestkit_forms.config import FormProcessorConfig
    from ingestkit_forms.models import FieldMapping, FormTemplate
    from ingestkit_forms.protocols import OCRBackend

logger = logging.getLogger("ingestkit_forms")


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _build_ocr_config(
    field_type: FieldType,
    engine_name: str,
    extraction_hint: str | None = None,
) -> str | None:
    """Build engine-specific OCR config string for a field type.

    Args:
        field_type: The field's data type.
        engine_name: OCR engine identifier ('tesseract' or 'paddleocr').
        extraction_hint: Optional hint from the field mapping.

    Returns:
        Engine-specific config string, or None for default config.
    """
    if engine_name == "tesseract":
        if field_type == FieldType.NUMBER:
            return "--psm 7 -c tessedit_char_whitelist=0123456789.-"
        elif field_type in (FieldType.TEXT, FieldType.DATE):
            return "--psm 7"
        return None
    elif engine_name == "paddleocr":
        # PaddleOCR config handled via its API, not CLI strings
        if field_type == FieldType.NUMBER:
            return "rec_char_type=EN"
        return None
    return None


def _crop_field_region(
    page_image: Image.Image,
    region: BoundingBox,
) -> Image.Image:
    """Crop a field region from a page image using normalized bbox.

    Converts normalized (0.0-1.0) coordinates to pixel coordinates
    and crops the region.

    Spec section 7.2 step 2a:
        px_x = region.x * page_width_px
        px_y = region.y * page_height_px
        px_w = region.width * page_width_px
        px_h = region.height * page_height_px
    """
    page_w, page_h = page_image.size
    px_x = int(region.x * page_w)
    px_y = int(region.y * page_h)
    px_w = int(region.width * page_w)
    px_h = int(region.height * page_h)

    # Clamp to image bounds
    px_x = max(0, min(px_x, page_w - 1))
    px_y = max(0, min(px_y, page_h - 1))
    right = min(px_x + px_w, page_w)
    bottom = min(px_y + px_h, page_h)

    return page_image.crop((px_x, px_y, right, bottom))


def _image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to PNG-encoded bytes for OCR backend."""
    buf = io.BytesIO()
    # Ensure image is in a mode that can be saved as PNG
    if image.mode == "1":
        image = image.convert("L")
    image.save(buf, format="PNG")
    return buf.getvalue()


def _post_process_value(
    raw_text: str,
    field_type: FieldType,
    extraction_hint: str | None = None,
) -> str:
    """Post-process raw OCR text per field type.

    - TEXT: strip whitespace
    - NUMBER: strip non-numeric chars (keep digits, '.', '-', ',')
    - DATE: strip whitespace, apply date hint if present
    """
    text = raw_text.strip()
    if field_type == FieldType.NUMBER:
        # Keep only digits, decimal point, minus, comma
        text = re.sub(r"[^\d.\-,]", "", text)
    elif field_type == FieldType.DATE:
        # Strip extra whitespace; date formatting left to caller
        text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------


class OCROverlayExtractor:
    """Extract field values by rendering document pages and OCR-ing each field region.

    Per spec section 7.2. All OCR access is via the OCRBackend protocol.
    """

    def __init__(
        self,
        ocr_backend: OCRBackend,
        config: FormProcessorConfig,
    ) -> None:
        self._ocr = ocr_backend
        self._config = config

    def extract(
        self,
        file_path: str,
        template: FormTemplate,
    ) -> list[ExtractedField]:
        """Extract all field values from a document using OCR overlay.

        Groups template fields by page_number, renders each page once,
        then processes each field on that page.

        Args:
            file_path: Path to the PDF or image file.
            template: Form template with field mappings.

        Returns:
            List of ExtractedField objects, one per template field.

        Raises:
            FormIngestError: If the entire extraction fails (e.g., file unreadable).
        """
        results: list[ExtractedField] = []

        # Group fields by page number
        fields_by_page: dict[int, list[FieldMapping]] = {}
        for field in template.fields:
            fields_by_page.setdefault(field.page_number, []).append(field)

        for page_num in sorted(fields_by_page.keys()):
            page_fields = fields_by_page[page_num]

            # Render page image (once per page for efficiency)
            try:
                page_image = get_page_image(
                    file_path, page_num, dpi=self._config.form_ocr_dpi
                )
            except FormIngestException:
                # Page rendering failed -- skip all fields on this page
                logger.warning(
                    "Page %d rendering failed for %s, skipping %d fields",
                    page_num,
                    file_path,
                    len(page_fields),
                )
                for field in page_fields:
                    results.append(
                        self._failed_field(
                            field,
                            FormErrorCode.E_FORM_EXTRACTION_FAILED,
                            f"Page {page_num} rendering failed",
                        )
                    )
                continue

            # Process each field on this page
            for field in page_fields:
                extracted = self._extract_single_field(page_image, field)
                results.append(extracted)

            # Release page image memory after processing all fields
            del page_image

        return results

    def _extract_single_field(
        self,
        page_image: Image.Image,
        field: FieldMapping,
    ) -> ExtractedField:
        """Extract a single field from a page image.

        Dispatches to field-type-specific logic: OCR for text types,
        fill/ink ratio for visual types.
        """
        warnings: list[str] = []

        if field.region is None:
            return self._failed_field(
                field,
                FormErrorCode.E_FORM_EXTRACTION_FAILED,
                "Field has no bounding box region",
            )

        # Crop field region
        crop = _crop_field_region(page_image, field.region)

        try:
            if field.field_type in (FieldType.TEXT, FieldType.NUMBER, FieldType.DATE):
                value, confidence, raw_value = self._extract_text_field(crop, field)
            elif field.field_type in (FieldType.CHECKBOX, FieldType.RADIO):
                value, confidence, raw_value = self._extract_checkbox_field(crop)
            elif field.field_type == FieldType.SIGNATURE:
                value, confidence, raw_value = self._extract_signature_field(crop)
            else:
                # DROPDOWN or unknown -- fail closed
                return self._failed_field(
                    field,
                    FormErrorCode.E_FORM_OCR_FAILED,
                    f"Unsupported field type for OCR: {field.field_type.value}",
                )
        except FormIngestException as e:
            logger.warning(
                "Field '%s' extraction failed: %s", field.field_name, e.message
            )
            return self._failed_field(field, e.code, e.message)
        except Exception as e:
            logger.warning(
                "Field '%s' extraction error: %s", field.field_name, str(e)
            )
            return self._failed_field(
                field,
                FormErrorCode.E_FORM_OCR_FAILED,
                str(e),
            )

        # Apply min confidence threshold
        if confidence < self._config.form_extraction_min_field_confidence:
            warnings.append(FormErrorCode.W_FORM_FIELD_LOW_CONFIDENCE.value)

        # Apply validation pattern (spec section 7.2 step 3)
        validation_passed: bool | None = None
        if field.validation_pattern and value is not None and isinstance(value, str):
            match_result = regex_match_with_timeout(
                field.validation_pattern, value, match_mode="match"
            )
            if match_result is None:
                # Regex timed out (ReDoS protection)
                warnings.append(FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value)
                validation_passed = False
                value = None
                confidence = 0.0
            elif match_result:
                validation_passed = True
            else:
                validation_passed = False
                warnings.append(FormErrorCode.W_FORM_FIELD_VALIDATION_FAILED.value)
                value = None
                confidence = 0.0

        # PII-safe logging
        if self._config.log_ocr_output and value is not None:
            logger.debug("OCR result for field '%s': %s", field.field_name, value)

        return ExtractedField(
            field_id=field.field_id,
            field_name=field.field_name,
            field_label=field.field_label,
            field_type=field.field_type,
            value=value,
            raw_value=raw_value,
            confidence=confidence,
            extraction_method="ocr_overlay",
            bounding_box=field.region,
            validation_passed=validation_passed,
            warnings=warnings,
        )

    def _extract_text_field(
        self,
        crop: Image.Image,
        field: FieldMapping,
    ) -> tuple[str | None, float, str | None]:
        """Extract a text-based field (TEXT, NUMBER, DATE) via OCR.

        Returns:
            (processed_value, confidence, raw_value)
        """
        preprocessed = preprocess_for_ocr(crop, field.field_type)
        image_bytes = _image_to_png_bytes(preprocessed)

        ocr_config = _build_ocr_config(
            field.field_type,
            self._ocr.engine_name(),
            field.extraction_hint,
        )

        try:
            result = self._ocr.ocr_region(
                image_bytes=image_bytes,
                language=self._config.form_ocr_language,
                config=ocr_config,
                timeout=float(self._config.form_ocr_per_field_timeout_seconds),
            )
        except TimeoutError:
            logger.warning(
                "OCR timeout for field '%s' after %ds",
                field.field_name,
                self._config.form_ocr_per_field_timeout_seconds,
            )
            return None, 0.0, None
        except Exception as e:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_OCR_FAILED,
                message=f"OCR failed for field '{field.field_name}': {e}",
                stage="ocr_overlay",
                field_name=field.field_name,
                recoverable=True,
            ) from e

        raw_text = result.text
        processed = _post_process_value(raw_text, field.field_type, field.extraction_hint)

        # Confidence = mean of char_confidences (spec section 7.2 step 2d)
        if result.char_confidences:
            confidence = sum(result.char_confidences) / len(result.char_confidences)
        else:
            confidence = result.confidence

        if not processed:
            return None, 0.0, raw_text

        return processed, confidence, raw_text

    def _extract_checkbox_field(
        self,
        crop: Image.Image,
    ) -> tuple[bool, float, str | None]:
        """Extract a CHECKBOX or RADIO field via fill ratio analysis.

        Spec section 7.2 step 2e:
            fill_ratio > checkbox_fill_threshold -> checked
            confidence = min(abs(fill_ratio - threshold) / threshold, 1.0)
        """
        preprocessed = preprocess_for_ocr(crop, FieldType.CHECKBOX)
        fill_ratio = compute_fill_ratio(preprocessed)
        threshold = self._config.checkbox_fill_threshold

        checked = fill_ratio > threshold
        confidence = (
            min(abs(fill_ratio - threshold) / threshold, 1.0) if threshold > 0 else 1.0
        )

        return checked, confidence, f"fill_ratio={fill_ratio:.4f}"

    def _extract_signature_field(
        self,
        crop: Image.Image,
    ) -> tuple[bool, float, str | None]:
        """Extract a SIGNATURE field via ink ratio analysis.

        Spec section 7.2 step 2f:
            ink_ratio > signature_ink_threshold -> signed
            confidence = min(abs(ink_ratio - threshold) / threshold, 1.0)
        """
        preprocessed = preprocess_for_ocr(crop, FieldType.SIGNATURE)
        ink_ratio = compute_ink_ratio(preprocessed)
        threshold = self._config.signature_ink_threshold

        signed = ink_ratio > threshold
        confidence = (
            min(abs(ink_ratio - threshold) / threshold, 1.0) if threshold > 0 else 1.0
        )

        return signed, confidence, f"ink_ratio={ink_ratio:.4f}"

    def _failed_field(
        self,
        field: FieldMapping,
        error_code: FormErrorCode,
        message: str,
    ) -> ExtractedField:
        """Create a fail-closed ExtractedField with value=None, confidence=0.0."""
        return ExtractedField(
            field_id=field.field_id,
            field_name=field.field_name,
            field_label=field.field_label,
            field_type=field.field_type,
            value=None,
            raw_value=None,
            confidence=0.0,
            extraction_method="ocr_overlay",
            bounding_box=field.region,
            validation_passed=None,
            warnings=[error_code.value, message],
        )
