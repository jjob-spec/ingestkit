"""Shared test fixtures for ingestkit-forms tests.

Provides mock backends, test configuration, and fixture placeholders
for form template, PDF, Excel, and image test files.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    BoundingBox,
    FieldMapping,
    FieldType,
    FormTemplate,
    SourceFormat,
)
from ingestkit_forms.protocols import OCRRegionResult, WidgetField


# ---------------------------------------------------------------------------
# Mock Backends
# ---------------------------------------------------------------------------


class MockPDFWidgetBackend:
    """Mock PDFWidgetBackend for testing NativePDFExtractor.

    Configurable via constructor: set widgets per page, and whether
    the PDF has form fields at all.
    """

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


class MockOCRBackend:
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


class MockLayoutFingerprinter:
    """Mock fingerprinter that returns pre-configured page fingerprints.

    Implements the LayoutFingerprinter protocol for unit testing without
    actual image rendering.
    """

    def __init__(self, pages: list[bytes] | None = None) -> None:
        self._pages = pages or []

    def compute_fingerprint(self, file_path: str) -> list[bytes]:
        return self._pages


class MockFormTemplateStore:
    """In-memory FormTemplateStore for unit testing.

    Implements the FormTemplateStore protocol without filesystem I/O.
    """

    def __init__(self) -> None:
        # {template_id: {version: FormTemplate}}
        self._templates: dict[str, dict[int, FormTemplate]] = {}
        self._deleted: dict[str, set[int]] = {}
        self._all_deleted: set[str] = set()

    def save_template(self, template: FormTemplate) -> None:
        tid = template.template_id
        if tid not in self._templates:
            self._templates[tid] = {}
        self._templates[tid][template.version] = template

    def get_template(
        self, template_id: str, version: int | None = None
    ) -> FormTemplate | None:
        versions = self._templates.get(template_id)
        if versions is None:
            return None
        if template_id in self._all_deleted:
            return None
        deleted = self._deleted.get(template_id, set())
        if version is None:
            for v in sorted(versions.keys(), reverse=True):
                if v not in deleted:
                    return versions[v]
            return None
        if version in deleted:
            return None
        return versions.get(version)

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
        active_only: bool = True,
    ) -> list[FormTemplate]:
        results: list[FormTemplate] = []
        for tid in self._templates:
            if active_only and tid in self._all_deleted:
                continue
            template = self.get_template(tid)
            if template is None:
                continue
            if tenant_id is not None and template.tenant_id != tenant_id:
                continue
            if source_format is not None and template.source_format.value != source_format:
                continue
            results.append(template)
        return results

    def list_versions(self, template_id: str) -> list[FormTemplate]:
        versions = self._templates.get(template_id)
        if versions is None:
            return []
        return sorted(versions.values(), key=lambda t: t.version, reverse=True)

    def delete_template(
        self, template_id: str, version: int | None = None
    ) -> None:
        if template_id not in self._templates:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=f"Template '{template_id}' not found",
                stage="template_store",
                recoverable=False,
            )
        if version is None:
            self._all_deleted.add(template_id)
        else:
            if version not in self._templates[template_id]:
                raise FormIngestException(
                    code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                    message=f"Template '{template_id}' version {version} not found",
                    stage="template_store",
                    recoverable=False,
                )
            if template_id not in self._deleted:
                self._deleted[template_id] = set()
            self._deleted[template_id].add(version)

    def get_all_fingerprints(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[tuple[str, str, int, bytes]]:
        templates = self.list_templates(
            tenant_id=tenant_id,
            source_format=source_format,
            active_only=True,
        )
        results: list[tuple[str, str, int, bytes]] = []
        for t in templates:
            if t.layout_fingerprint is not None:
                results.append(
                    (t.template_id, t.name, t.version, t.layout_fingerprint)
                )
        return results


# ---------------------------------------------------------------------------
# Factory Helpers
# ---------------------------------------------------------------------------


def make_widget(
    field_name: str = "field_1",
    field_value: str | None = "some value",
    field_type: str = "text",
    x: float = 0.1,
    y: float = 0.1,
    width: float = 0.3,
    height: float = 0.05,
    page: int = 0,
) -> WidgetField:
    """Factory for WidgetField test instances."""
    return WidgetField(
        field_name=field_name,
        field_value=field_value,
        field_type=field_type,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
        page=page,
    )


def make_field_mapping(
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
    """Factory for FieldMapping test instances."""
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


def make_template(
    fields: list[FieldMapping] | None = None,
    name: str = "Test Template",
    source_format: SourceFormat = SourceFormat.PDF,
    page_count: int = 1,
    layout_fingerprint: bytes | None = None,
    template_id: str | None = None,
    tenant_id: str | None = None,
    version: int = 1,
) -> FormTemplate:
    """Factory for FormTemplate test instances."""
    if fields is None:
        fields = [make_field_mapping()]
    kwargs: dict = dict(
        name=name,
        source_format=source_format,
        page_count=page_count,
        fields=fields,
        layout_fingerprint=layout_fingerprint,
        version=version,
    )
    if template_id is not None:
        kwargs["template_id"] = template_id
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    return FormTemplate(**kwargs)


def make_uniform_page(value: int, rows: int = 20, cols: int = 16) -> list[list[int]]:
    """Create a uniform grid page filled with a single quantization value (0-3)."""
    return [[value] * cols for _ in range(rows)]


def make_page_bytes(page: list[list[int]]) -> bytes:
    """Convert a single page grid to bytes."""
    result = bytearray()
    for row in page:
        result.extend(row)
    return bytes(result)


def make_fingerprint_bytes(
    pages: list[list[list[int]]],
) -> bytes:
    """Serialize per-page grid matrices into concatenated fingerprint bytes."""
    result = bytearray()
    for page in pages:
        for row in page:
            result.extend(row)
    return bytes(result)


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def form_config():
    """Return a FormProcessorConfig with all defaults."""
    return FormProcessorConfig()


# ---------------------------------------------------------------------------
# Mock Backend Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_template_store():
    """Mock FormTemplateStore for testing."""
    return MockFormTemplateStore()


@pytest.fixture()
def mock_fingerprinter():
    """Mock LayoutFingerprinter for testing."""
    return MockLayoutFingerprinter()


@pytest.fixture()
def mock_ocr_backend():
    """Mock OCRBackend for testing."""
    return MockOCRBackend()


@pytest.fixture()
def mock_pdf_widget_backend():
    """Mock PDFWidgetBackend with no widgets (override in tests)."""
    return MockPDFWidgetBackend()


# ---------------------------------------------------------------------------
# Image Fixtures for Fingerprinting
# ---------------------------------------------------------------------------


@pytest.fixture()
def blank_page_image():
    """Return a 1200x1600 white image (simulates a blank letter-size page at 150 DPI)."""
    return Image.new("L", (1200, 1600), color=255)


@pytest.fixture()
def form_like_image():
    """Return a synthetic image with structural elements (lines, boxes) for fingerprint testing."""
    img = Image.new("L", (1200, 1600), color=255)
    draw = ImageDraw.Draw(img)
    # Draw header bar
    draw.rectangle([0, 0, 1200, 100], fill=0)
    # Draw field boxes
    for i in range(5):
        y = 200 + i * 200
        draw.rectangle([50, y, 600, y + 40], outline=0, width=2)
        draw.rectangle([650, y, 1150, y + 40], outline=0, width=2)
    # Draw footer line
    draw.line([0, 1500, 1200, 1500], fill=0, width=3)
    return img


@pytest.fixture()
def sample_image_file(tmp_path, form_like_image):
    """Save form_like_image to a .png file and return the path string."""
    path = tmp_path / "sample_form.png"
    form_like_image.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Test File Fixtures (Placeholders)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pdf_form(tmp_path):
    """Generate a sample fillable PDF form for testing.

    TODO: Implement once extractors are built.
    """
    return tmp_path / "sample_form.pdf"


@pytest.fixture()
def sample_excel_form(tmp_path):
    """Generate a sample Excel form for testing.

    TODO: Implement once extractors are built.
    """
    return tmp_path / "sample_form.xlsx"


@pytest.fixture()
def sample_scanned_form(tmp_path):
    """Generate a sample scanned form image for testing.

    TODO: Implement once extractors are built.
    """
    return tmp_path / "sample_form.png"
