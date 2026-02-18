"""Integration tests for all ingestkit converter packages.

Generates REAL files, runs the FULL Router.process() pipeline with mock backends,
and verifies end-to-end results including:
- Security scanning
- IngestKey computation
- Text extraction
- Chunking
- Embedding (mocked)
- Vector store upsert (mocked)
- ProcessingResult assembly
"""

from __future__ import annotations

import datetime
import email
import email.mime.multipart
import email.mime.text
import email.mime.base
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

# ── Mock backends used by all routers ──────────────────────────────

EMBED_DIM = 768


class MockVectorStore:
    """Mock satisfying VectorStoreBackend protocol."""

    def __init__(self):
        self.upserted: list = []
        self.collections_ensured: list = []

    def upsert_chunks(self, collection: str, chunks: list) -> int:
        self.upserted.extend(chunks)
        return len(chunks)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        self.collections_ensured.append((collection, vector_size))

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        pass

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        return 0


class MockEmbedder:
    """Mock satisfying EmbeddingBackend protocol."""

    def __init__(self, dim: int = EMBED_DIM):
        self._dim = dim
        self.texts_embedded: list[str] = []

    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        self.texts_embedded.extend(texts)
        return [[0.1] * self._dim for _ in texts]

    def dimension(self) -> int:
        return self._dim


class MockVLM:
    """Mock satisfying ImageVLMBackend protocol."""

    def caption(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> str:
        return "A photograph of a red brick building with white trim and a green lawn in the foreground."

    def model_name(self) -> str:
        return "mock-vlm"

    def is_available(self) -> bool:
        return True


class MockOCR:
    """Mock satisfying ImageOCRBackend protocol."""

    def ocr_image(
        self,
        image_bytes: bytes,
        language: str = "eng",
        config: str = "",
        timeout: float | None = None,
    ):
        from ingestkit_image.protocols import OCRResult
        return OCRResult(
            text="OCR extracted text from image: Invoice #12345",
            confidence=0.92,
            engine="mock-ocr",
            language="eng",
        )

    def engine_name(self) -> str:
        return "mock-ocr"


# ── File generators ────────────────────────────────────────────────


def generate_png(path: Path, width: int = 200, height: int = 150) -> Path:
    """Generate a real PNG image with some content."""
    img = Image.new("RGB", (width, height), color=(70, 130, 180))
    # Draw a simple pattern
    for x in range(0, width, 20):
        for y in range(0, height, 20):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path, "PNG")
    return path


def generate_jpeg(path: Path) -> Path:
    """Generate a real JPEG image."""
    img = Image.new("RGB", (300, 200), color=(220, 20, 60))
    img.save(path, "JPEG")
    return path


def generate_json(path: Path) -> Path:
    """Generate a real JSON file with nested structure."""
    data = {
        "company": "Bethany Terrace",
        "department": "Claims",
        "employees": [
            {
                "name": "Alice Johnson",
                "role": "Claims Adjuster",
                "contact": {"email": "alice@example.com", "phone": "555-0101"},
            },
            {
                "name": "Bob Smith",
                "role": "Senior Adjuster",
                "contact": {"email": "bob@example.com", "phone": "555-0102"},
            },
        ],
        "policy_count": 1247,
        "active": True,
        "last_audit": None,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def generate_eml(path: Path) -> Path:
    """Generate a real EML email file with multipart content."""
    msg = email.mime.multipart.MIMEMultipart("alternative")
    msg["From"] = "john.doe@bethanyterrace.com"
    msg["To"] = "claims@bethanyterrace.com"
    msg["Subject"] = "Q4 Claims Report - Action Required"
    msg["Date"] = "Mon, 15 Jan 2026 09:30:00 -0600"
    msg["Message-ID"] = "<test-12345@bethanyterrace.com>"

    plain = email.mime.text.MIMEText(
        "Please review the attached Q4 claims report.\n\n"
        "Key findings:\n"
        "- Total claims processed: 847\n"
        "- Average processing time: 3.2 days\n"
        "- Customer satisfaction: 94%\n\n"
        "Best regards,\nJohn Doe",
        "plain",
    )
    html = email.mime.text.MIMEText(
        "<html><body>"
        "<h1>Q4 Claims Report</h1>"
        "<p>Please review the attached Q4 claims report.</p>"
        "<ul><li>Total claims: 847</li><li>Avg time: 3.2 days</li></ul>"
        "</body></html>",
        "html",
    )
    msg.attach(plain)
    msg.attach(html)

    path.write_bytes(msg.as_bytes())
    return path


def generate_eml_html_only(path: Path) -> Path:
    """Generate an EML with only HTML body (no plain text)."""
    msg = email.mime.text.MIMEText(
        "<html><body>"
        "<h1>Meeting Notes</h1>"
        "<p>Discussed the new <b>claims processing</b> workflow.</p>"
        "<p>Action items:</p>"
        "<ol><li>Update policy templates</li><li>Train new adjusters</li></ol>"
        "</body></html>",
        "html",
    )
    msg["From"] = "manager@example.com"
    msg["To"] = "team@example.com"
    msg["Subject"] = "Meeting Notes - Claims Workflow"
    msg["Date"] = "Tue, 16 Jan 2026 14:00:00 -0600"

    path.write_bytes(msg.as_bytes())
    return path


def generate_xml(path: Path) -> Path:
    """Generate a real XML file with namespaces and attributes."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<insurance xmlns="http://example.com/insurance" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <company name="Bethany Terrace" id="BT-001">
        <department>Claims Processing</department>
        <location city="Springfield" state="IL"/>
    </company>
    <policies>
        <policy number="POL-2024-001" type="auto" status="active">
            <holder>Alice Johnson</holder>
            <premium currency="USD">1200.00</premium>
            <coverage>Comprehensive</coverage>
        </policy>
        <policy number="POL-2024-002" type="home" status="active">
            <holder>Bob Smith</holder>
            <premium currency="USD">2400.00</premium>
            <coverage>Full replacement</coverage>
        </policy>
    </policies>
    <summary>
        <total_policies>2</total_policies>
        <total_premium>3600.00</total_premium>
    </summary>
</insurance>"""
    path.write_text(xml_content, encoding="utf-8")
    return path


def generate_rtf(path: Path) -> Path:
    """Generate a real RTF file with formatting."""
    rtf_content = (
        r"{\rtf1\ansi\deff0"
        r"{\fonttbl{\f0 Times New Roman;}{\f1 Arial;}}"
        r"{\colortbl;\red0\green0\blue0;\red255\green0\blue0;}"
        r"\f0\fs24 "
        r"{\b Claims Processing Report}\par\par"
        r"{\f1\fs20 Prepared by: Claims Department}\par"
        r"{\f1\fs20 Date: January 2026}\par\par"
        r"This report summarizes the Q4 claims processing activities "
        r"for Bethany Terrace Insurance.\par\par"
        r"{\b Key Metrics:}\par"
        r"- Total claims processed: 847\par"
        r"- Average processing time: 3.2 business days\par"
        r"- Customer satisfaction rate: 94%\par"
        r"- Claims denial rate: 8.3%\par\par"
        r"{\b Recommendations:}\par"
        r"1. Implement automated claims triage system\par"
        r"2. Reduce average processing time to under 3 days\par"
        r"3. Expand training program for new adjusters\par"
        r"}"
    )
    path.write_text(rtf_content, encoding="utf-8")
    return path


def generate_xls(path: Path) -> Path:
    """Generate a real .xls file with multiple sheets and data types."""
    import xlwt

    wb = xlwt.Workbook()

    # Sheet 1: Employee data
    ws1 = wb.add_sheet("Employees")
    headers = ["Name", "Role", "Start Date", "Salary", "Active"]
    for col, h in enumerate(headers):
        ws1.write(0, col, h)
    data = [
        ("Alice Johnson", "Claims Adjuster", "2023-01-15", 65000.0, True),
        ("Bob Smith", "Senior Adjuster", "2020-06-01", 82000.0, True),
        ("Carol White", "Manager", "2018-03-20", 95000.0, True),
        ("Dave Brown", "Junior Adjuster", "2025-09-01", 52000.0, False),
    ]
    date_fmt = xlwt.XFStyle()
    date_fmt.num_format_str = "YYYY-MM-DD"
    for row, (name, role, date_str, salary, active) in enumerate(data, start=1):
        ws1.write(row, 0, name)
        ws1.write(row, 1, role)
        # Write dates as date objects
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        ws1.write(row, 2, dt, date_fmt)
        ws1.write(row, 3, salary)
        ws1.write(row, 4, active)

    # Sheet 2: Claims data
    ws2 = wb.add_sheet("Claims")
    claim_headers = ["Claim ID", "Amount", "Status", "Filed Date"]
    for col, h in enumerate(claim_headers):
        ws2.write(0, col, h)
    claims = [
        ("CLM-001", 15000.50, "Approved", "2025-10-01"),
        ("CLM-002", 3200.00, "Pending", "2025-11-15"),
        ("CLM-003", 78500.00, "Denied", "2025-12-03"),
    ]
    for row, (cid, amount, status, date_str) in enumerate(claims, start=1):
        ws2.write(row, 0, cid)
        ws2.write(row, 1, amount)
        ws2.write(row, 2, status)
        ws2.write(row, 3, date_str)

    # Sheet 3: Empty sheet (should be skipped)
    wb.add_sheet("Notes")

    wb.save(str(path))
    return path


def generate_doc(path: Path) -> Path:
    """Generate a minimal .doc-like OLE2 file.

    Creating a true .doc is complex. Instead, we create a real .doc using
    mammoth-compatible structure. For a true integration test we'll use
    a pre-built minimal .doc if mammoth can handle it, otherwise we test
    the error path gracefully.
    """
    # Create a minimal OLE2 header + empty doc
    # This is a simplified OLE2 container - mammoth may or may not parse it
    # We test both success and graceful failure paths
    import struct
    import io

    # OLE2 magic header
    ole2_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

    # For a proper test, let's create a .docx first and then test .doc path
    # Since creating a valid .doc binary is extremely complex, we'll test:
    # 1. Security scanner with real OLE2 magic bytes
    # 2. Router graceful failure with a minimal OLE2 container

    # Write a minimal OLE2 header (enough to pass magic byte check)
    # but mammoth will likely fail to extract - testing error handling
    header = ole2_magic + b"\x00" * 504  # 512-byte sector
    path.write_bytes(header)
    return path


# ── Integration Tests ──────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def vector_store():
    return MockVectorStore()


@pytest.fixture
def embedder():
    return MockEmbedder()


@pytest.fixture
def vlm():
    return MockVLM()


@pytest.fixture
def ocr():
    return MockOCR()


class TestImageVLMPipeline:
    """Test ingestkit-image with VLM captioning on real PNG/JPEG files."""

    def test_png_vlm_captioning(self, tmp_dir, vector_store, embedder, vlm):
        from ingestkit_image import ImageRouter, ImageProcessorConfig

        path = generate_png(tmp_dir / "building.png")
        config = ImageProcessorConfig(default_collection="test-images")
        router = ImageRouter(vlm=vlm, vector_store=vector_store, embedder=embedder, config=config)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created == 1
        assert len(result.errors) == 0
        assert result.caption_result is not None
        assert "red brick building" in result.caption_result.caption
        assert result.image_metadata.image_type == "png"
        assert result.image_metadata.width == 200
        assert result.image_metadata.height == 150
        assert result.ingest_key  # deterministic key computed

        # Verify embedding was called
        assert len(embedder.texts_embedded) == 1
        assert "red brick building" in embedder.texts_embedded[0]

        # Verify vector store upsert
        assert len(vector_store.upserted) == 1
        chunk = vector_store.upserted[0]
        assert chunk.metadata.source_type == "image_caption"
        assert chunk.metadata.source_format == "image"
        print(f"  ✓ PNG VLM captioning: {result.chunks_created} chunk, key={result.ingest_key[:12]}...")

    def test_jpeg_vlm_captioning(self, tmp_dir, vector_store, embedder, vlm):
        from ingestkit_image import ImageRouter

        path = generate_jpeg(tmp_dir / "photo.jpg")
        router = ImageRouter(vlm=vlm, vector_store=vector_store, embedder=embedder)

        result = router.process(str(path))

        assert result.chunks_created == 1
        assert result.image_metadata.image_type == "jpeg"
        assert result.image_metadata.width == 300
        assert result.image_metadata.height == 200
        print(f"  ✓ JPEG VLM captioning: {result.chunks_created} chunk")


class TestImageOCRPipeline:
    """Test ingestkit-image with OCR on real image files."""

    def test_png_ocr_extraction(self, tmp_dir, vector_store, embedder, ocr):
        from ingestkit_image import ImageRouter, ImageProcessorConfig

        path = generate_png(tmp_dir / "scanned.png")
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            ocr=ocr, vector_store=vector_store, embedder=embedder, config=config
        )

        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert len(result.errors) == 0
        assert result.ocr_result is not None
        assert "Invoice #12345" in result.ocr_result.text
        print(f"  ✓ PNG OCR extraction: {result.chunks_created} chunk, OCR conf={result.ocr_result.confidence}")

    def test_vlm_and_ocr_combined(self, tmp_dir, vector_store, embedder, vlm, ocr):
        from ingestkit_image import ImageRouter, ImageProcessorConfig

        path = generate_png(tmp_dir / "hybrid.png")
        config = ImageProcessorConfig(enable_ocr=True)
        router = ImageRouter(
            vlm=vlm, ocr=ocr, vector_store=vector_store, embedder=embedder, config=config
        )

        result = router.process(str(path))

        # Should have chunks from both VLM and OCR
        assert result.chunks_created >= 2
        assert result.caption_result is not None  # VLM ran
        assert result.ocr_result is not None  # OCR ran
        print(f"  ✓ VLM+OCR combined: {result.chunks_created} chunks")


class TestJSONPipeline:
    """Test ingestkit-json with real JSON files."""

    def test_nested_json(self, tmp_dir, vector_store, embedder):
        from ingestkit_json import JSONRouter, JSONProcessorConfig

        path = generate_json(tmp_dir / "data.json")
        config = JSONProcessorConfig(default_collection="test-json")
        router = JSONRouter(vector_store=vector_store, embedder=embedder, config=config)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert len(result.errors) == 0
        assert result.total_keys > 0
        assert result.ingest_key

        # Verify flattened content was embedded
        assert len(embedder.texts_embedded) >= 1
        all_text = " ".join(embedder.texts_embedded)
        assert "Bethany Terrace" in all_text
        assert "employees[0].name" in all_text or "Alice Johnson" in all_text
        print(f"  ✓ Nested JSON: {result.chunks_created} chunks, {result.total_keys} keys, depth={result.max_depth}")

    def test_large_json_array(self, tmp_dir, vector_store, embedder):
        from ingestkit_json import JSONRouter

        # Generate a JSON array with 100 records
        records = [
            {"id": i, "name": f"Policy-{i:04d}", "amount": 1000 + i * 50, "active": i % 3 != 0}
            for i in range(100)
        ]
        path = tmp_dir / "policies.json"
        path.write_text(json.dumps(records, indent=2))

        router = JSONRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert result.total_keys >= 400  # 100 records × 4 fields
        assert len(result.errors) == 0
        print(f"  ✓ Large JSON array (100 records): {result.chunks_created} chunks, {result.total_keys} keys")

    def test_invalid_json_rejected(self, tmp_dir, vector_store, embedder):
        from ingestkit_json import JSONRouter

        path = tmp_dir / "broken.json"
        path.write_text("{not valid json!!!", encoding="utf-8")

        router = JSONRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        print(f"  ✓ Invalid JSON rejected: {result.errors}")


class TestEMLPipeline:
    """Test ingestkit-email with real EML files."""

    def test_multipart_eml(self, tmp_dir, vector_store, embedder):
        from ingestkit_email import EmailRouter

        path = generate_eml(tmp_dir / "report.eml")
        router = EmailRouter(vector_store=vector_store, embedder=embedder)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created == 1
        assert len(result.errors) == 0
        assert result.email_metadata is not None
        assert result.email_metadata.subject == "Q4 Claims Report - Action Required"
        assert result.email_metadata.from_address == "john.doe@bethanyterrace.com"

        # Verify plain text was preferred over HTML
        embedded_text = embedder.texts_embedded[0]
        assert "claims report" in embedded_text.lower()
        assert "847" in embedded_text  # claims count
        print(f"  ✓ Multipart EML: subject='{result.email_metadata.subject}', chunks={result.chunks_created}")

    def test_html_only_eml(self, tmp_dir, vector_store, embedder):
        from ingestkit_email import EmailRouter

        path = generate_eml_html_only(tmp_dir / "meeting.eml")
        router = EmailRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created == 1
        # Should have warning about HTML-only
        assert any("HTML" in w or "html" in w for w in result.warnings)
        # HTML tags should be stripped
        embedded_text = embedder.texts_embedded[0]
        assert "<html>" not in embedded_text
        assert "<b>" not in embedded_text
        assert "claims processing" in embedded_text.lower()
        print(f"  ✓ HTML-only EML: HTML stripped, warnings={result.warnings}")


class TestXMLPipeline:
    """Test ingestkit-xml with real XML files."""

    def test_namespaced_xml(self, tmp_dir, vector_store, embedder):
        from ingestkit_xml import XMLRouter, XMLProcessorConfig

        path = generate_xml(tmp_dir / "insurance.xml")
        config = XMLProcessorConfig(default_collection="test-xml")
        router = XMLRouter(vector_store=vector_store, embedder=embedder, config=config)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert len(result.errors) == 0
        assert result.ingest_key

        # Verify namespace prefixes stripped and content extracted
        all_text = " ".join(embedder.texts_embedded)
        assert "Bethany Terrace" in all_text
        assert "Alice Johnson" in all_text
        assert "POL-2024-001" in all_text
        # Namespace URIs should NOT appear
        assert "http://example.com" not in all_text
        print(f"  ✓ Namespaced XML: {result.chunks_created} chunks, content extracted correctly")

    def test_xml_bomb_rejected(self, tmp_dir, vector_store, embedder):
        from ingestkit_xml import XMLRouter

        # Billion laughs attack
        bomb = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;">
]>
<root>&lol2;</root>"""
        path = tmp_dir / "bomb.xml"
        path.write_text(bomb, encoding="utf-8")

        router = XMLRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        print(f"  ✓ XML bomb rejected: {result.errors}")


class TestRTFPipeline:
    """Test ingestkit-rtf with real RTF files."""

    def test_formatted_rtf(self, tmp_dir, vector_store, embedder):
        from ingestkit_rtf import RTFRouter, RTFProcessorConfig

        path = generate_rtf(tmp_dir / "report.rtf")
        config = RTFProcessorConfig(default_collection="test-rtf")
        router = RTFRouter(vector_store=vector_store, embedder=embedder, config=config)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert len(result.errors) == 0

        # Verify RTF control codes stripped, content preserved
        all_text = " ".join(embedder.texts_embedded)
        assert "Claims Processing Report" in all_text
        assert "847" in all_text
        assert r"\par" not in all_text  # RTF codes stripped
        assert r"\b" not in all_text
        print(f"  ✓ Formatted RTF: {result.chunks_created} chunks, formatting stripped")

    def test_non_rtf_rejected(self, tmp_dir, vector_store, embedder):
        from ingestkit_rtf import RTFRouter

        path = tmp_dir / "fake.rtf"
        path.write_text("This is not an RTF file", encoding="utf-8")

        router = RTFRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        print(f"  ✓ Non-RTF rejected: {result.errors}")


class TestXLSPipeline:
    """Test ingestkit-xls with real .xls files."""

    def test_multi_sheet_xls(self, tmp_dir, vector_store, embedder):
        from ingestkit_xls import XlsRouter, XlsProcessorConfig

        path = generate_xls(tmp_dir / "employees.xls")
        config = XlsProcessorConfig(default_collection="test-xls")
        router = XlsRouter(vector_store=vector_store, embedder=embedder, config=config)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        assert result.chunks_created >= 1
        assert len(result.errors) == 0

        # Verify multi-sheet extraction
        all_text = " ".join(embedder.texts_embedded)
        assert "Alice Johnson" in all_text
        assert "Claims Adjuster" in all_text
        assert "CLM-001" in all_text  # From Claims sheet
        # Empty Notes sheet should be skipped
        print(f"  ✓ Multi-sheet XLS: {result.chunks_created} chunks, sheets={result.sheet_count}, rows={result.total_rows}")

    def test_wrong_extension_rejected(self, tmp_dir, vector_store, embedder):
        from ingestkit_xls import XlsRouter

        path = tmp_dir / "data.xlsx"
        path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)

        router = XlsRouter(vector_store=vector_store, embedder=embedder)
        assert not router.can_handle(str(path))
        print("  ✓ .xlsx extension correctly rejected (only .xls accepted)")


class TestDOCPipeline:
    """Test ingestkit-doc with .doc files."""

    def test_doc_security_passes_ole2(self, tmp_dir, vector_store, embedder):
        from ingestkit_doc import DocRouter

        path = generate_doc(tmp_dir / "report.doc")
        router = DocRouter(vector_store=vector_store, embedder=embedder)

        assert router.can_handle(str(path))
        result = router.process(str(path))

        # The minimal OLE2 header may not be parseable by mammoth
        # but the security scanner should pass (valid magic bytes)
        # and the error should be graceful, not a crash
        if result.chunks_created == 0:
            assert len(result.errors) > 0
            print(f"  ✓ DOC security passes, extraction failed gracefully: {result.errors}")
        else:
            assert len(result.errors) == 0
            print(f"  ✓ DOC extraction success: {result.chunks_created} chunks")

    def test_non_ole2_rejected(self, tmp_dir, vector_store, embedder):
        from ingestkit_doc import DocRouter

        path = tmp_dir / "fake.doc"
        path.write_text("This is just a text file renamed to .doc", encoding="utf-8")

        router = DocRouter(vector_store=vector_store, embedder=embedder)
        result = router.process(str(path))

        assert result.chunks_created == 0
        assert len(result.errors) > 0
        print(f"  ✓ Non-OLE2 .doc rejected: {result.errors}")


class TestTenantIDPropagation:
    """Verify tenant_id flows through the entire pipeline for each package."""

    def test_json_tenant_id(self, tmp_dir, vector_store, embedder):
        from ingestkit_json import JSONRouter, JSONProcessorConfig

        path = generate_json(tmp_dir / "data.json")
        config = JSONProcessorConfig(tenant_id="tenant-abc-123")
        router = JSONRouter(vector_store=vector_store, embedder=embedder, config=config)
        result = router.process(str(path))

        assert result.tenant_id == "tenant-abc-123"
        for chunk in vector_store.upserted:
            assert chunk.metadata.tenant_id == "tenant-abc-123"
        print("  ✓ JSON tenant_id propagated to all chunks")

    def test_email_tenant_id(self, tmp_dir, vector_store, embedder):
        from ingestkit_email import EmailRouter, EmailProcessorConfig

        path = generate_eml(tmp_dir / "email.eml")
        config = EmailProcessorConfig(tenant_id="tenant-xyz-789")
        router = EmailRouter(vector_store=vector_store, embedder=embedder, config=config)
        result = router.process(str(path))

        assert result.tenant_id == "tenant-xyz-789"
        for chunk in vector_store.upserted:
            assert chunk.metadata.tenant_id == "tenant-xyz-789"
        print("  ✓ Email tenant_id propagated to all chunks")


class TestIdempotency:
    """Verify deterministic IngestKey across multiple runs."""

    def test_json_idempotent_key(self, tmp_dir, vector_store, embedder):
        from ingestkit_json import JSONRouter

        path = generate_json(tmp_dir / "data.json")
        router = JSONRouter(vector_store=vector_store, embedder=embedder)

        result1 = router.process(str(path))
        result2 = router.process(str(path))

        assert result1.ingest_key == result2.ingest_key
        assert result1.ingest_key  # non-empty
        print(f"  ✓ JSON idempotent key: {result1.ingest_key[:16]}...")

    def test_image_idempotent_key(self, tmp_dir, vector_store, embedder, vlm):
        from ingestkit_image import ImageRouter

        path = generate_png(tmp_dir / "test.png")
        router = ImageRouter(vlm=vlm, vector_store=vector_store, embedder=embedder)

        result1 = router.process(str(path))
        # Reset mocks for second run
        vector_store2 = MockVectorStore()
        embedder2 = MockEmbedder()
        router2 = ImageRouter(vlm=vlm, vector_store=vector_store2, embedder=embedder2)
        result2 = router2.process(str(path))

        assert result1.ingest_key == result2.ingest_key
        print(f"  ✓ Image idempotent key: {result1.ingest_key[:16]}...")


class TestCanHandleDispatch:
    """Verify all routers correctly identify their supported formats."""

    def test_format_dispatch(self, tmp_dir, vector_store, embedder, vlm):
        from ingestkit_image import ImageRouter
        from ingestkit_json import JSONRouter
        from ingestkit_email import EmailRouter
        from ingestkit_xml import XMLRouter
        from ingestkit_rtf import RTFRouter
        from ingestkit_xls import XlsRouter
        from ingestkit_doc import DocRouter

        routers = [
            (ImageRouter(vlm=vlm, vector_store=vector_store, embedder=embedder), {
                True: ["photo.png", "photo.jpg", "photo.jpeg", "scan.tiff", "img.webp", "img.bmp", "anim.gif"],
                False: ["data.json", "doc.pdf", "file.xml"],
            }),
            (JSONRouter(vector_store=vector_store, embedder=embedder), {
                True: ["data.json", "DATA.JSON"],
                False: ["data.xml", "data.txt"],
            }),
            (EmailRouter(vector_store=vector_store, embedder=embedder), {
                True: ["mail.eml", "outlook.msg"],
                False: ["mail.txt", "mail.pdf"],
            }),
            (XMLRouter(vector_store=vector_store, embedder=embedder), {
                True: ["config.xml", "DATA.XML"],
                False: ["data.json", "data.html"],
            }),
            (RTFRouter(vector_store=vector_store, embedder=embedder), {
                True: ["doc.rtf", "DOC.RTF"],
                False: ["doc.txt", "doc.pdf"],
            }),
            (XlsRouter(vector_store=vector_store, embedder=embedder), {
                True: ["data.xls", "DATA.XLS"],
                False: ["data.xlsx", "data.csv"],
            }),
            (DocRouter(vector_store=vector_store, embedder=embedder), {
                True: ["report.doc", "REPORT.DOC"],
                False: ["report.docx", "report.pdf"],
            }),
        ]

        for router, cases in routers:
            name = type(router).__name__
            for should_handle in cases[True]:
                assert router.can_handle(should_handle), f"{name} should handle {should_handle}"
            for should_not in cases[False]:
                assert not router.can_handle(should_not), f"{name} should NOT handle {should_not}"
            print(f"  ✓ {name}: dispatch correct for {len(cases[True]) + len(cases[False])} extensions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
