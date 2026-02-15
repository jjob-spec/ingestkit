"""Shared fixtures for ingestkit-pdf tests.

Provides MockLLMBackend, factory helpers for DocumentProfile / PageProfile,
programmatic PDF fixtures, and common pytest fixtures used across test modules.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Frame,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.inspector import PDFInspector
from ingestkit_pdf.models import (
    DocumentMetadata,
    DocumentProfile,
    ExtractionQuality,
    PageProfile,
    PageType,
)


# ---------------------------------------------------------------------------
# Sentinel Constants for MockLLMBackend
# ---------------------------------------------------------------------------

_SENTINEL_TIMEOUT = "__TIMEOUT__"
_SENTINEL_CONNECTION_ERROR = "__CONNECTION_ERROR__"
_SENTINEL_MALFORMED_JSON = "__MALFORMED_JSON__"

# Password used by the encrypted_pdf fixture
ENCRYPTED_PDF_PASSWORD = "testpass123"


# ---------------------------------------------------------------------------
# Mock LLM Backend
# ---------------------------------------------------------------------------


class MockLLMBackend:
    """Mock LLM backend for testing PDFLLMClassifier.

    Supports configurable responses via a list of return values or
    exceptions.  Each call to ``classify()`` pops the next item from
    the response queue, allowing tests to simulate retry sequences.

    NOTE: ``enqueue_classify`` and ``enqueue_generate`` share the same
    underlying ``_responses`` queue, matching the existing single-queue
    design.  If you need to interleave classify and generate calls,
    enqueue responses in the order the calls will be made.
    """

    def __init__(
        self,
        responses: list[dict | str | Exception] | None = None,
    ) -> None:
        self._responses: list[dict | str | Exception] = list(responses or [])
        self.calls: list[dict] = []  # records all calls for assertion

    # -- Core protocol methods -----------------------------------------------

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise RuntimeError("MockLLMBackend: no more responses configured")
        response = self._responses.pop(0)
        # Sentinel handling (checked before isinstance Exception)
        if response == _SENTINEL_TIMEOUT:
            raise TimeoutError("MockLLMBackend simulated timeout")
        if response == _SENTINEL_CONNECTION_ERROR:
            raise ConnectionError("MockLLMBackend simulated connection error")
        if response == _SENTINEL_MALFORMED_JSON:
            return {"raw": "<<<not json>>>"}
        if isinstance(response, Exception):
            raise response
        return response

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise RuntimeError("MockLLMBackend: no more responses configured")
        response = self._responses.pop(0)
        # Sentinel handling (checked before isinstance Exception)
        if response == _SENTINEL_TIMEOUT:
            raise TimeoutError("MockLLMBackend simulated timeout")
        if response == _SENTINEL_CONNECTION_ERROR:
            raise ConnectionError("MockLLMBackend simulated connection error")
        if isinstance(response, Exception):
            raise response
        return str(response)

    # -- Convenience enqueue methods -----------------------------------------

    def enqueue_classify(self, *responses: dict | str | Exception) -> None:
        """Append responses to the shared queue for classify() calls."""
        self._responses.extend(responses)

    def enqueue_generate(self, *responses: dict | str | Exception) -> None:
        """Append responses to the shared queue for generate() calls."""
        self._responses.extend(responses)

    def enqueue_timeout(self) -> None:
        """Enqueue a sentinel that causes the next call to raise TimeoutError."""
        self._responses.append(_SENTINEL_TIMEOUT)

    def enqueue_connection_error(self) -> None:
        """Enqueue a sentinel that causes the next call to raise ConnectionError."""
        self._responses.append(_SENTINEL_CONNECTION_ERROR)

    # -- Assertion helpers ---------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of classify/generate calls made."""
        return len(self.calls)

    def assert_called_with_model(self, model: str) -> None:
        """Assert that at least one call used the specified model."""
        assert any(c["model"] == model for c in self.calls), (
            f"Expected call with model={model!r}, got: {[c['model'] for c in self.calls]}"
        )


# ---------------------------------------------------------------------------
# Mock Vector Store Backend
# ---------------------------------------------------------------------------


class MockVectorStoreBackend:
    """Mock vector store for testing processors."""

    def __init__(self) -> None:
        self.collections_ensured: list[tuple[str, int]] = []
        self.upserted: list[tuple[str, list]] = []
        self.payload_indices: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, list[str]]] = []
        self._errors: dict[str, Exception] = {}

    def upsert_chunks(self, collection: str, chunks: list) -> int:
        if "upsert" in self._errors:
            err = self._errors.pop("upsert")
            raise err
        self.upserted.append((collection, list(chunks)))
        return len(chunks)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        if "ensure" in self._errors:
            err = self._errors.pop("ensure")
            raise err
        self.collections_ensured.append((collection, vector_size))

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        self.payload_indices.append((collection, field, field_type))

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        self.deleted.append((collection, ids))
        return len(ids)

    # -- Error injection -----------------------------------------------------

    def fail_next_upsert(self, error: Exception | None = None) -> None:
        """Make the next upsert_chunks() call raise an error."""
        self._errors["upsert"] = error or ConnectionError("MockVectorStore simulated error")

    def fail_next_ensure(self, error: Exception | None = None) -> None:
        """Make the next ensure_collection() call raise an error."""
        self._errors["ensure"] = error or ConnectionError("MockVectorStore simulated error")

    # -- Assertion helpers ---------------------------------------------------

    @property
    def total_chunks_upserted(self) -> int:
        """Total number of chunks across all upsert calls."""
        return sum(len(chunks) for _, chunks in self.upserted)


# ---------------------------------------------------------------------------
# Mock Embedding Backend
# ---------------------------------------------------------------------------


class MockEmbeddingBackend:
    """Mock embedding backend for testing processors."""

    def __init__(self, dim: int = 768) -> None:
        self._dimension = dim
        self.calls: list[list[str]] = []
        self._error_on_next: Exception | None = None

    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        if self._error_on_next is not None:
            err = self._error_on_next
            self._error_on_next = None
            raise err
        self.calls.append(list(texts))
        return [[0.1] * self._dimension for _ in texts]

    def dimension(self) -> int:
        return self._dimension

    # -- Error injection -----------------------------------------------------

    def fail_next_embed(self, error: Exception | None = None) -> None:
        """Make the next embed() call raise an error."""
        self._error_on_next = error or TimeoutError("MockEmbeddingBackend simulated timeout")

    # -- Assertion helpers ---------------------------------------------------

    @property
    def total_texts_embedded(self) -> int:
        """Total number of texts across all embed calls."""
        return sum(len(batch) for batch in self.calls)


# ---------------------------------------------------------------------------
# Mock Structured DB Backend
# ---------------------------------------------------------------------------


class MockStructuredDBBackend:
    """Mock structured DB backend for testing processors."""

    def __init__(self) -> None:
        self.tables: dict[str, Any] = {}
        self.dropped: list[str] = []
        self._error_on_next: Exception | None = None

    def create_table_from_dataframe(self, table_name: str, df: Any) -> None:
        if self._error_on_next is not None:
            err = self._error_on_next
            self._error_on_next = None
            raise err
        self.tables[table_name] = df

    def drop_table(self, table_name: str) -> None:
        self.tables.pop(table_name, None)
        self.dropped.append(table_name)

    def table_exists(self, table_name: str) -> bool:
        return table_name in self.tables

    def get_table_schema(self, table_name: str) -> dict:
        if table_name not in self.tables:
            return {}
        df = self.tables[table_name]
        return {col: str(df[col].dtype) for col in df.columns}

    def get_connection_uri(self) -> str:
        return "sqlite:///:memory:"

    # -- Error injection -----------------------------------------------------

    def fail_next_create(self, error: Exception | None = None) -> None:
        """Make the next create_table_from_dataframe() call raise an error."""
        self._error_on_next = error or ConnectionError(
            "MockStructuredDBBackend simulated error"
        )


# ---------------------------------------------------------------------------
# Factory Helpers
# ---------------------------------------------------------------------------


def _make_extraction_quality(**overrides: Any) -> ExtractionQuality:
    """Build an ExtractionQuality with sensible defaults."""
    defaults: dict[str, Any] = dict(
        printable_ratio=0.95,
        avg_words_per_page=300.0,
        pages_with_text=1,
        total_pages=1,
        extraction_method="pdfminer",
    )
    defaults.update(overrides)
    return ExtractionQuality(**defaults)


def _make_page_profile(**overrides: Any) -> PageProfile:
    """Build a PageProfile with sensible text-page defaults."""
    defaults: dict[str, Any] = dict(
        page_number=1,
        text_length=1500,
        word_count=300,
        image_count=0,
        image_coverage_ratio=0.0,
        table_count=0,
        font_count=3,
        font_names=["Arial", "Times", "Courier"],
        has_form_fields=False,
        is_multi_column=False,
        page_type=PageType.TEXT,
        extraction_quality=_make_extraction_quality(),
    )
    defaults.update(overrides)
    return PageProfile(**defaults)


def _make_document_profile(
    pages: list[PageProfile] | None = None,
    **overrides: Any,
) -> DocumentProfile:
    """Build a DocumentProfile from a list of PageProfiles."""
    if pages is None:
        pages = [_make_page_profile()]

    # Auto-compute page_type_distribution from pages
    distribution: dict[str, int] = {}
    for p in pages:
        key = p.page_type.value
        distribution[key] = distribution.get(key, 0) + 1

    defaults: dict[str, Any] = dict(
        file_path="/tmp/test.pdf",
        file_size_bytes=102400,
        page_count=len(pages),
        content_hash="a" * 64,
        metadata=DocumentMetadata(
            creator="TestApp",
            pdf_version="1.7",
            page_count=len(pages),
            file_size_bytes=102400,
        ),
        pages=pages,
        page_type_distribution=distribution,
        detected_languages=["en"],
        has_toc=False,
        toc_entries=None,
        overall_quality=_make_extraction_quality(
            pages_with_text=len(pages),
            total_pages=len(pages),
        ),
        security_warnings=[],
    )
    defaults.update(overrides)
    return DocumentProfile(**defaults)


def _valid_response(
    type_: str = "text_native",
    confidence: float = 0.85,
    reasoning: str = "Digital PDF with extractable text throughout.",
    page_types: list[dict[str, Any]] | None = None,
) -> dict:
    """Build a valid LLM response dict."""
    d: dict[str, Any] = {
        "type": type_,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    if page_types is not None:
        d["page_types"] = page_types
    return d


# ---------------------------------------------------------------------------
# Pytest Fixtures — Mock Backends
# ---------------------------------------------------------------------------


@pytest.fixture()
def pdf_config() -> PDFProcessorConfig:
    return PDFProcessorConfig()


@pytest.fixture()
def pdf_inspector(pdf_config: PDFProcessorConfig) -> PDFInspector:
    return PDFInspector(pdf_config)


@pytest.fixture()
def document_profile() -> DocumentProfile:
    return _make_document_profile()


@pytest.fixture()
def mock_structured_db() -> MockStructuredDBBackend:
    return MockStructuredDBBackend()


@pytest.fixture()
def mock_vector_store() -> MockVectorStoreBackend:
    return MockVectorStoreBackend()


@pytest.fixture()
def mock_embedder() -> MockEmbeddingBackend:
    return MockEmbeddingBackend()


@pytest.fixture()
def mock_llm() -> MockLLMBackend:
    return MockLLMBackend()


# ---------------------------------------------------------------------------
# Pytest Fixtures — Programmatic PDF Generation
# ---------------------------------------------------------------------------


@pytest.fixture()
def text_native_pdf(tmp_path: Path) -> Path:
    """Multi-page PDF with headings, paragraphs, and page numbers.

    Produces a clean digital PDF with extractable text (3 pages).
    """
    path = tmp_path / "text_native.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    _page_width, _page_height = letter

    chapters = [
        (
            "Chapter 1: Introduction",
            [
                "This document provides an overview of the project goals and methodology.",
                "The research was conducted over a period of six months with regular reviews.",
                "Key findings are summarized in the following chapters for stakeholder review.",
            ],
        ),
        (
            "Chapter 2: Methods",
            [
                "Data was collected from multiple sources including surveys and interviews.",
                "Statistical analysis was performed using standard regression techniques.",
                "All results were validated through cross-reference with existing literature.",
            ],
        ),
        (
            "Chapter 3: Results",
            [
                "The analysis revealed significant improvements across all measured metrics.",
                "Response rates exceeded expectations at ninety-two percent overall completion.",
                "Detailed tables and figures are provided in the appendices for reference.",
            ],
        ),
    ]

    for page_num, (heading, paragraphs) in enumerate(chapters, start=1):
        # Heading
        c.setFont("Helvetica-Bold", 18)
        c.drawString(72, 700, heading)

        # Body paragraphs
        c.setFont("Helvetica", 11)
        y = 660
        for para in paragraphs:
            c.drawString(72, y, para)
            y -= 20

        # Page number footer
        c.setFont("Helvetica", 9)
        c.drawString(280, 40, f"Page {page_num} of 3")

        c.showPage()

    c.save()
    return path


@pytest.fixture()
def scanned_pdf(tmp_path: Path) -> Path:
    """PDF where pages contain only raster images (no text layer).

    Simulates a scanned document — 2 pages of rendered text as images.
    """
    path = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    page_width, page_height = letter

    page_texts = [
        [
            "Scanned Document - Page 1",
            "This is the first page of a scanned document.",
            "It contains text rendered as an image only.",
        ],
        [
            "Scanned Document - Page 2",
            "The second page has different content.",
            "No extractable text layer exists in this PDF.",
        ],
    ]

    dpi = 150
    img_w = int(8.5 * dpi)
    img_h = int(11 * dpi)

    for texts in page_texts:
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        # Use default font (guaranteed available)
        y_pos = 150
        for line in texts:
            draw.text((100, y_pos), line, fill="black")
            y_pos += 40

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        c.drawImage(
            ImageReader(buf),
            0,
            0,
            width=page_width,
            height=page_height,
        )
        c.showPage()

    c.save()
    return path


@pytest.fixture()
def complex_pdf(tmp_path: Path) -> Path:
    """PDF with tables, multi-column layout, and mixed content.

    Page 1: Title heading + data table (5 cols x 8 rows).
    Page 2: Two-column text section + smaller summary table.
    """
    path = tmp_path / "complex.pdf"

    # -- Page 1: title + data table ------------------------------------------
    def _page1(c: canvas.Canvas, doc: Any) -> None:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "Quarterly Employee Report")
        c.setFont("Helvetica", 9)
        c.drawString(72, 30, f"Confidential - Page {doc.page}")

    # -- Page 2: footer only -------------------------------------------------
    def _page2(c: canvas.Canvas, doc: Any) -> None:
        c.setFont("Helvetica", 9)
        c.drawString(72, 30, f"Confidential - Page {doc.page}")

    styles = getSampleStyleSheet()

    # Build page templates
    page_width, page_height = letter
    margin = 72  # 1 inch

    # Page 1 template: single full-width frame below the title
    frame_p1 = Frame(
        margin,
        margin + 20,
        page_width - 2 * margin,
        page_height - 2 * margin - 80,
        id="page1_frame",
    )
    template_p1 = PageTemplate(id="page1", frames=[frame_p1], onPage=_page1)

    # Page 2 template: two-column layout
    col_width = (page_width - 2 * margin - 20) / 2
    frame_left = Frame(
        margin,
        margin + 20,
        col_width,
        page_height - 2 * margin - 40,
        id="left_col",
    )
    frame_right = Frame(
        margin + col_width + 20,
        margin + 20,
        col_width,
        page_height - 2 * margin - 40,
        id="right_col",
    )
    template_p2 = PageTemplate(
        id="page2",
        frames=[frame_left, frame_right],
        onPage=_page2,
    )

    doc = SimpleDocTemplate(str(path), pagesize=letter)
    doc.addPageTemplates([template_p1, template_p2])

    story: list[Any] = []

    # Data table
    table_data = [
        ["ID", "Name", "Department", "Salary", "Start Date"],
        ["001", "Alice Johnson", "Engineering", "$95,000", "2021-03-15"],
        ["002", "Bob Smith", "Marketing", "$78,000", "2020-07-01"],
        ["003", "Carol Davis", "Engineering", "$102,000", "2019-11-20"],
        ["004", "David Wilson", "Finance", "$88,000", "2022-01-10"],
        ["005", "Eva Martinez", "Marketing", "$82,000", "2021-08-22"],
        ["006", "Frank Lee", "Engineering", "$97,000", "2020-04-05"],
        ["007", "Grace Kim", "Finance", "$91,000", "2022-06-15"],
    ]

    t = Table(table_data, colWidths=[40, 100, 90, 70, 80])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#D9E2F3")]),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 24))

    # Force page break and switch to two-column template
    from reportlab.platypus import NextPageTemplate, PageBreak

    story.append(NextPageTemplate("page2"))
    story.append(PageBreak())

    # Two-column text content
    left_text = (
        "The engineering department continues to show strong growth in both "
        "headcount and average compensation. Three new hires were added in Q3, "
        "bringing the total team size to twenty-four engineers across frontend, "
        "backend, and infrastructure roles."
    )
    right_text = (
        "Marketing has stabilized after the reorganization completed last quarter. "
        "Campaign performance metrics are up fifteen percent year-over-year, and "
        "the new digital strategy is expected to drive additional growth in Q4."
    )

    story.append(Paragraph(left_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Summary table in right column
    summary_data = [
        ["Department", "Headcount", "Avg Salary"],
        ["Engineering", "24", "$98,000"],
        ["Marketing", "12", "$80,000"],
        ["Finance", "8", "$89,500"],
    ]
    t2 = Table(summary_data, colWidths=[80, 60, 70])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )

    # FrameBreak to move to right column
    from reportlab.platypus import FrameBreak

    story.append(FrameBreak())
    story.append(Paragraph(right_text, styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(t2)

    doc.build(story)
    return path


@pytest.fixture()
def encrypted_pdf(tmp_path: Path) -> Path:
    """Password-protected PDF (AES-256 encryption).

    Use ``ENCRYPTED_PDF_PASSWORD`` constant (``testpass123``) to open.
    """
    path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    tw = fitz.TextWriter(page.rect)
    tw.append((100, 700), "This document is encrypted and confidential.")
    tw.write_text(page)
    perm = fitz.PDF_PERM_ACCESSIBILITY | fitz.PDF_PERM_PRINT
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=ENCRYPTED_PDF_PASSWORD,
        owner_pw="ownerpass456",
        permissions=perm,
    )
    doc.close()
    return path


@pytest.fixture()
def garbled_pdf(tmp_path: Path) -> Path:
    """PDF with garbled/non-printable text simulating CIDFont encoding issues.

    When extracted via PyMuPDF, the text should have a low printable ratio
    (below 0.5), exercising quality scoring and ``E_PARSE_GARBLED`` detection.
    """
    path = tmp_path / "garbled.pdf"
    doc = fitz.open()

    for page_num in range(2):
        page = doc.new_page()
        tw = fitz.TextWriter(page.rect)
        # Build garbled text: mix of replacement chars, control chars, and
        # minimal printable text so the PDF is structurally valid.
        garbled_parts = [
            "\ufffd\ufffd\ufffd",
            "\x01\x02\x03\x04\x05",
            "ab",
            "\ufffd\ufffd",
            "\x06\x07\x08",
            "c",
            "\ufffd\ufffd\ufffd\ufffd",
            "\x0e\x0f\x10\x11",
            "d",
            "\ufffd\ufffd\ufffd",
            "\x12\x13\x14\x15\x16",
        ]
        garbled_text = "".join(garbled_parts)
        try:
            tw.append((72, 700), garbled_text)
            tw.write_text(page)
        except Exception:
            # Fallback: if fitz rejects the characters, insert via page
            # insertion which is more permissive with raw text.
            page.insert_text((72, 700), garbled_text)

    doc.save(str(path))
    doc.close()
    return path
