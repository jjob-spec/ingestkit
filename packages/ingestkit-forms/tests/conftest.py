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
    ExtractedField,
    FieldMapping,
    FieldType,
    FormExtractionResult,
    FormTemplate,
    SourceFormat,
)
from ingestkit_forms.protocols import OCRRegionResult, VLMFieldResult, WidgetField


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


class MockVLMBackend:
    """Mock VLMBackend for testing VLM fallback extraction.

    Configurable: default return value, confidence, availability,
    and optional per-call overrides. Tracks call count and arguments.
    """

    def __init__(
        self,
        default_value: str | bool = "extracted_value",
        default_confidence: float = 0.85,
        available: bool = True,
        model: str = "mock-vlm",
        raise_timeout: bool = False,
        raise_error: bool = False,
    ) -> None:
        self._default_value = default_value
        self._default_confidence = default_confidence
        self._available = available
        self._model = model
        self._raise_timeout = raise_timeout
        self._raise_error = raise_error
        self.call_count = 0
        self.call_args: list[dict] = []

    def extract_field(
        self,
        image_bytes: bytes,
        field_type: str,
        field_name: str,
        extraction_hint: str | None = None,
        timeout: float | None = None,
    ) -> VLMFieldResult:
        self.call_count += 1
        self.call_args.append({
            "field_type": field_type,
            "field_name": field_name,
            "extraction_hint": extraction_hint,
            "timeout": timeout,
        })
        if self._raise_timeout:
            raise TimeoutError("VLM timeout")
        if self._raise_error:
            raise RuntimeError("VLM backend error")
        return VLMFieldResult(
            value=self._default_value,
            confidence=self._default_confidence,
            model=self._model,
            prompt_tokens=100,
            completion_tokens=20,
        )

    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self._available


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


class MockFormDBBackend:
    """In-memory FormDBBackend for testing output writers.

    Implements the FormDBBackend protocol with simple dict storage.
    Tracks all calls for assertion in tests.
    """

    def __init__(self, *, fail_on_call: int = -1) -> None:
        self._tables: dict[str, dict[str, str]] = {}  # table -> {col: type}
        self._rows: dict[str, list[dict]] = {}  # table -> [row_dict, ...]
        self.execute_sql_calls: list[tuple[str, tuple | None]] = []
        self.delete_rows_calls: list[tuple[str, str, list[str]]] = []
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def execute_sql(self, sql: str, params: tuple | None = None) -> None:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"Mock DB failure on call {self._call_count}")
        self.execute_sql_calls.append((sql, params))

        sql_upper = sql.strip().upper()
        if sql_upper.startswith("CREATE TABLE"):
            self._handle_create_table(sql)
        elif sql_upper.startswith("ALTER TABLE"):
            self._handle_alter_table(sql)
        elif sql_upper.startswith("INSERT OR REPLACE"):
            self._handle_insert(sql, params)

    def _handle_create_table(self, sql: str) -> None:
        import re

        match = re.match(r"CREATE TABLE IF NOT EXISTS (\S+)\s*\((.+)\)", sql)
        if not match:
            return
        table_name = match.group(1)
        cols_str = match.group(2)
        schema: dict[str, str] = {}
        for col_def in cols_str.split(","):
            col_def = col_def.strip()
            parts = col_def.split()
            if len(parts) >= 2:
                col_name = parts[0]
                col_type = parts[1]
                schema[col_name] = col_type
        self._tables[table_name] = schema
        self._rows[table_name] = []

    def _handle_alter_table(self, sql: str) -> None:
        import re

        match = re.match(
            r"ALTER TABLE (\S+) ADD COLUMN (\S+) (\S+)", sql, re.IGNORECASE
        )
        if not match:
            return
        table_name = match.group(1)
        col_name = match.group(2)
        col_type = match.group(3)
        if table_name in self._tables:
            self._tables[table_name][col_name] = col_type

    def _handle_insert(self, sql: str, params: tuple | None) -> None:
        import re

        match = re.match(
            r"INSERT OR REPLACE INTO (\S+)\s*\((.+?)\)\s*VALUES\s*\((.+)\)",
            sql,
            re.IGNORECASE,
        )
        if not match or params is None:
            return
        table_name = match.group(1)
        columns = [c.strip() for c in match.group(2).split(",")]
        row = dict(zip(columns, params))

        if table_name not in self._rows:
            self._rows[table_name] = []

        # Upsert by _form_id
        existing_idx = None
        for i, existing_row in enumerate(self._rows[table_name]):
            if existing_row.get("_form_id") == row.get("_form_id"):
                existing_idx = i
                break
        if existing_idx is not None:
            self._rows[table_name][existing_idx] = row
        else:
            self._rows[table_name].append(row)

    def get_table_columns(self, table_name: str) -> list[str]:
        if table_name not in self._tables:
            return []
        return list(self._tables[table_name].keys())

    def delete_rows(self, table_name: str, column: str, values: list[str]) -> int:
        self.delete_rows_calls.append((table_name, column, values))
        if table_name not in self._rows:
            return 0
        original_count = len(self._rows[table_name])
        self._rows[table_name] = [
            row for row in self._rows[table_name] if row.get(column) not in values
        ]
        return original_count - len(self._rows[table_name])

    def table_exists(self, table_name: str) -> bool:
        return table_name in self._tables

    def get_connection_uri(self) -> str:
        return "mock://memory"


class MockVectorStoreBackend:
    """In-memory VectorStoreBackend for testing chunk writers.

    Tracks all calls for assertion in tests.
    """

    def __init__(self, *, fail_on_upsert: bool = False) -> None:
        self._collections: dict[str, dict[str, object]] = {}
        self.upsert_calls: list[tuple[str, list]] = []
        self.delete_calls: list[tuple[str, list[str]]] = []
        self._fail_on_upsert = fail_on_upsert

    def upsert_chunks(self, collection: str, chunks: list) -> int:
        if self._fail_on_upsert:
            raise RuntimeError("Mock vector upsert failure")
        self.upsert_calls.append((collection, chunks))
        if collection not in self._collections:
            self._collections[collection] = {}
        for chunk in chunks:
            self._collections[collection][chunk.id] = chunk
        return len(chunks)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        if collection not in self._collections:
            self._collections[collection] = {}

    def create_payload_index(
        self, collection: str, field: str, field_type: str
    ) -> None:
        pass  # no-op

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        self.delete_calls.append((collection, ids))
        if collection not in self._collections:
            return 0
        count = 0
        for id_ in ids:
            if id_ in self._collections[collection]:
                del self._collections[collection][id_]
                count += 1
        return count


class MockEmbeddingBackend:
    """Mock EmbeddingBackend returning deterministic vectors.

    Tracks embed calls for assertion in tests.
    """

    def __init__(
        self,
        dim: int = 768,
        *,
        fail_on_call: int = -1,
    ) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []
        self._fail_on_call = fail_on_call
        self._call_count = 0

    def embed(
        self, texts: list[str], timeout: float | None = None
    ) -> list[list[float]]:
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"Mock embed failure on call {self._call_count}")
        self.embed_calls.append(texts)
        return [[0.1] * self._dim for _ in texts]

    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Extraction Result Factory
# ---------------------------------------------------------------------------


def make_extracted_field(
    field_id: str = "f1",
    field_name: str = "employee_name",
    field_label: str = "Employee Name",
    field_type: FieldType = FieldType.TEXT,
    value: str | bool | float | None = "John Doe",
    confidence: float = 0.9,
    extraction_method: str = "ocr_overlay",
) -> ExtractedField:
    """Factory for ExtractedField test instances."""
    return ExtractedField(
        field_id=field_id,
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        value=value,
        confidence=confidence,
        extraction_method=extraction_method,
    )


def make_extraction_result(
    fields: list[ExtractedField] | None = None,
    template_name: str = "Test Template",
    template_id: str = "tmpl-1",
    template_version: int = 1,
    source_uri: str = "/forms/test.pdf",
    source_format: str = "pdf",
    overall_confidence: float = 0.85,
    extraction_method: str = "ocr_overlay",
    match_method: str = "auto_detect",
    form_id: str = "form-001",
) -> FormExtractionResult:
    """Factory for FormExtractionResult test instances."""
    if fields is None:
        fields = [make_extracted_field()]
    return FormExtractionResult(
        form_id=form_id,
        template_id=template_id,
        template_name=template_name,
        template_version=template_version,
        source_uri=source_uri,
        source_format=source_format,
        fields=fields,
        overall_confidence=overall_confidence,
        extraction_method=extraction_method,
        match_method=match_method,
        pages_processed=1,
        extraction_duration_seconds=1.5,
    )


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
    field_id: str | None = None,
    extraction_hint: str | None = None,
) -> FieldMapping:
    """Factory for FieldMapping test instances."""
    kwargs: dict = dict(
        field_name=field_name,
        field_label=field_label,
        field_type=field_type,
        page_number=page_number,
        region=BoundingBox(x=x, y=y, width=width, height=height),
        validation_pattern=validation_pattern,
        required=required,
        options=options,
        extraction_hint=extraction_hint,
    )
    if field_id is not None:
        kwargs["field_id"] = field_id
    return FieldMapping(**kwargs)


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


@pytest.fixture()
def mock_vlm_backend():
    """Mock VLMBackend for testing VLM fallback."""
    return MockVLMBackend()


@pytest.fixture()
def vlm_enabled_config():
    """FormProcessorConfig with VLM enabled."""
    return FormProcessorConfig(
        form_vlm_enabled=True,
        form_vlm_fallback_threshold=0.4,
        form_vlm_max_fields_per_document=10,
        form_vlm_timeout_seconds=15,
    )


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


# ---------------------------------------------------------------------------
# Output Writer Mock Backend Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_form_db():
    """Mock FormDBBackend for testing output writers."""
    return MockFormDBBackend()


@pytest.fixture()
def mock_vector_store():
    """Mock VectorStoreBackend for testing chunk writers."""
    return MockVectorStoreBackend()


@pytest.fixture()
def mock_embedder():
    """Mock EmbeddingBackend for testing chunk writers."""
    return MockEmbeddingBackend()


@pytest.fixture()
def output_config():
    """FormProcessorConfig tuned for fast output writer tests (no real sleeps)."""
    return FormProcessorConfig(
        backend_max_retries=1,
        backend_backoff_base=0.0,
        backend_timeout_seconds=5.0,
        tenant_id="test-tenant",
    )
