"""Backend protocols for the ingestkit-forms pipeline.

Defines form-specific protocols (FormTemplateStore, OCRBackend,
PDFWidgetBackend, VLMBackend) and their result models.
Re-exports shared protocols from ingestkit_core.

See SPEC section 15.3 for authoritative definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ingestkit_core.protocols import (
    EmbeddingBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)
from ingestkit_forms.models import BoundingBox

if TYPE_CHECKING:
    from ingestkit_forms.models import FormTemplate

__all__ = [
    # Re-exported from ingestkit-core
    "VectorStoreBackend",
    "StructuredDBBackend",
    "EmbeddingBackend",
    # Form-specific protocols
    "FormTemplateStore",
    "OCRBackend",
    "PDFWidgetBackend",
    "VLMBackend",
    # Result models
    "OCRRegionResult",
    "WidgetField",
    "VLMFieldResult",
]


# ---------------------------------------------------------------------------
# Result Models (defined before protocols that reference them)
# ---------------------------------------------------------------------------


class OCRRegionResult(BaseModel):
    """Result of OCR on a single field region."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    char_confidences: list[float] | None = Field(
        default=None,
        description="Per-character confidence values, if available.",
    )
    engine: str


class WidgetField(BaseModel):
    """A single form widget extracted from a PDF."""

    field_name: str
    field_value: str | None
    field_type: str  # "text", "checkbox", "radio", "dropdown", "listbox"
    bbox: BoundingBox  # Normalized 0.0-1.0 coordinates
    page: int


class VLMFieldResult(BaseModel):
    """Result of VLM extraction on a single field region."""

    value: str | bool | None
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# Form-Specific Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FormTemplateStore(Protocol):
    """Interface for form template persistence.

    Concrete implementations might use: filesystem (JSON/YAML files),
    SQLite, PostgreSQL, or any key-value store.
    """

    def save_template(self, template: FormTemplate) -> None:
        """Persist a template (insert or update)."""
        ...

    def get_template(
        self, template_id: str, version: int | None = None
    ) -> FormTemplate | None:
        """Retrieve a template by ID. None if not found.

        If version is None, returns the latest version.
        """
        ...

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
        active_only: bool = True,
    ) -> list[FormTemplate]:
        """List templates matching the filters."""
        ...

    def list_versions(self, template_id: str) -> list[FormTemplate]:
        """List all versions of a template, ordered by version descending."""
        ...

    def delete_template(
        self, template_id: str, version: int | None = None
    ) -> None:
        """Soft-delete a template or specific version."""
        ...

    def get_all_fingerprints(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[tuple[str, str, int, bytes]]:
        """Return (template_id, name, version, fingerprint) for all active templates.

        Used by the matcher for efficient batch comparison.
        """
        ...


@runtime_checkable
class OCRBackend(Protocol):
    """Interface for OCR engines used in form field extraction.

    Abstracts Tesseract vs. PaddleOCR (or any future engine).
    """

    def ocr_region(
        self,
        image_bytes: bytes,
        language: str = "en",
        config: str | None = None,
        timeout: float | None = None,
    ) -> OCRRegionResult:
        """Run OCR on a cropped image region.

        Args:
            image_bytes: PNG-encoded bytes of the cropped region.
            language: OCR language code.
            config: Engine-specific configuration string
                (e.g., Tesseract --psm and --oem flags).
            timeout: Per-field timeout in seconds.

        Returns:
            OCRRegionResult with text, confidence, and character-level details.
        """
        ...

    def engine_name(self) -> str:
        """Return the name of the OCR engine (e.g., 'tesseract', 'paddleocr')."""
        ...


@runtime_checkable
class PDFWidgetBackend(Protocol):
    """Interface for extracting form widgets from fillable PDFs.

    Abstracts PyMuPDF (AGPL) vs. pdfplumber+pypdf (MIT/BSD).
    See spec section 7.1.1 for licensing governance.
    """

    def extract_widgets(
        self,
        file_path: str,
        page: int,
    ) -> list[WidgetField]:
        """Extract all form widgets from the specified page.

        Args:
            file_path: Path to the PDF file.
            page: 0-indexed page number.

        Returns:
            List of WidgetField objects with field_name, field_value,
            field_type, and bounding box in normalized coordinates.
        """
        ...

    def has_form_fields(self, file_path: str) -> bool:
        """Check whether the PDF contains any fillable form fields."""
        ...

    def engine_name(self) -> str:
        """Return the backend name (e.g., 'pymupdf', 'pdfplumber')."""
        ...


@runtime_checkable
class VLMBackend(Protocol):
    """Interface for Vision-Language Model field extraction.

    Abstracts Ollama/Qwen2.5-VL vs. vLLM vs. llama.cpp.
    Only used when form_vlm_enabled=True and OCR confidence is
    below form_vlm_fallback_threshold.
    """

    def extract_field(
        self,
        image_bytes: bytes,
        field_type: str,
        field_name: str,
        extraction_hint: str | None = None,
        timeout: float | None = None,
    ) -> VLMFieldResult:
        """Run VLM extraction on a cropped field image.

        Args:
            image_bytes: PNG-encoded bytes of the cropped field region
                (with padding for context).
            field_type: Expected field type ('text', 'number', 'date',
                'checkbox', 'radio', 'signature', 'dropdown').
            field_name: Human-readable field name for prompt context.
            extraction_hint: Optional hint (e.g., 'date_format:MM/DD/YYYY').
            timeout: Per-field timeout in seconds.

        Returns:
            VLMFieldResult with extracted value and confidence.
        """
        ...

    def model_name(self) -> str:
        """Return the VLM model identifier (e.g., 'qwen2.5-vl:7b')."""
        ...

    def is_available(self) -> bool:
        """Check whether the VLM backend is reachable."""
        ...
