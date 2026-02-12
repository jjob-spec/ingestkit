"""Tests for ingestkit_pdf.security — pre-flight security scanner.

All test PDFs are generated programmatically (no binary fixtures).
"""

from __future__ import annotations

import io

import fitz  # PyMuPDF
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode
from ingestkit_pdf.security import PDFSecurityScanner


# ---------------------------------------------------------------------------
# Fixtures — programmatic PDF generation
# ---------------------------------------------------------------------------


def _make_simple_pdf(num_pages: int = 1, text: str = "Hello World") -> bytes:
    """Generate a simple PDF with N pages using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for i in range(num_pages):
        c.drawString(100, 700, f"{text} - page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


def _write_pdf(tmp_path, content: bytes, name: str = "test.pdf") -> str:
    """Write PDF bytes to a temp file and return the path."""
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


@pytest.fixture
def scanner():
    """Scanner with default config."""
    return PDFSecurityScanner(PDFProcessorConfig())


@pytest.fixture
def simple_pdf(tmp_path):
    """Path to a valid single-page PDF."""
    return _write_pdf(tmp_path, _make_simple_pdf())


# ---------------------------------------------------------------------------
# Valid PDF Tests
# ---------------------------------------------------------------------------


class TestValidPDF:
    def test_accepts_valid_pdf(self, scanner, simple_pdf):
        metadata, errors = scanner.scan(simple_pdf)
        fatal = [e for e in errors if e.code.value.startswith("E_")]
        assert len(fatal) == 0
        assert metadata.page_count == 1
        assert metadata.file_size_bytes > 0

    def test_multi_page_pdf(self, scanner, tmp_path):
        path = _write_pdf(tmp_path, _make_simple_pdf(num_pages=5))
        metadata, errors = scanner.scan(path)
        fatal = [e for e in errors if e.code.value.startswith("E_")]
        assert len(fatal) == 0
        assert metadata.page_count == 5

    def test_metadata_extracted(self, scanner, simple_pdf):
        metadata, _ = scanner.scan(simple_pdf)
        assert metadata.pdf_version is not None
        assert metadata.is_encrypted is False
        assert metadata.needs_password is False
        assert metadata.is_linearized is False


# ---------------------------------------------------------------------------
# Magic Bytes Tests
# ---------------------------------------------------------------------------


class TestMagicBytes:
    def test_invalid_magic_bytes(self, scanner, tmp_path):
        path = _write_pdf(tmp_path, b"NOT A PDF FILE CONTENT", name="bad.pdf")
        metadata, errors = scanner.scan(path)
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.E_SECURITY_INVALID_PDF

    def test_empty_file(self, scanner, tmp_path):
        path = _write_pdf(tmp_path, b"", name="empty.pdf")
        metadata, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_SECURITY_INVALID_PDF for e in errors)

    def test_truncated_header(self, scanner, tmp_path):
        path = _write_pdf(tmp_path, b"%PD", name="truncated.pdf")
        metadata, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_SECURITY_INVALID_PDF for e in errors)


# ---------------------------------------------------------------------------
# File Size Tests
# ---------------------------------------------------------------------------


class TestFileSize:
    def test_oversized_file(self, tmp_path):
        # Config with tiny limit to avoid generating actual large files
        config = PDFProcessorConfig(max_file_size_mb=0)
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf())
        _, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_SECURITY_TOO_LARGE for e in errors)

    def test_oversized_with_override(self, tmp_path):
        config = PDFProcessorConfig(
            max_file_size_mb=0,
            max_file_size_override_reason="TICKET-100: test override",
        )
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf())
        _, errors = scanner.scan(path)
        # Should NOT have fatal error
        fatal = [e for e in errors if e.code == ErrorCode.E_SECURITY_TOO_LARGE]
        assert len(fatal) == 0
        # Should have warning
        overrides = [e for e in errors if e.code == ErrorCode.W_SECURITY_OVERRIDE]
        assert len(overrides) >= 1
        assert "TICKET-100" in overrides[0].message


# ---------------------------------------------------------------------------
# Page Count Tests
# ---------------------------------------------------------------------------


class TestPageCount:
    def test_too_many_pages(self, tmp_path):
        config = PDFProcessorConfig(max_page_count=2)
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf(num_pages=5))
        _, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_SECURITY_TOO_MANY_PAGES for e in errors)

    def test_page_count_at_limit(self, tmp_path):
        config = PDFProcessorConfig(max_page_count=5)
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf(num_pages=5))
        _, errors = scanner.scan(path)
        fatal = [e for e in errors if e.code == ErrorCode.E_SECURITY_TOO_MANY_PAGES]
        assert len(fatal) == 0

    def test_page_count_override(self, tmp_path):
        config = PDFProcessorConfig(
            max_page_count=2,
            max_page_count_override_reason="TICKET-200: large doc set",
        )
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf(num_pages=5))
        _, errors = scanner.scan(path)
        fatal = [e for e in errors if e.code == ErrorCode.E_SECURITY_TOO_MANY_PAGES]
        assert len(fatal) == 0
        overrides = [e for e in errors if e.code == ErrorCode.W_SECURITY_OVERRIDE]
        assert len(overrides) >= 1

    def test_zero_pages(self, tmp_path):
        """PDF with zero pages should fail with E_PARSE_EMPTY."""
        # Craft a minimal valid PDF with zero pages (PyMuPDF refuses to save one)
        minimal_pdf = (
            b"%PDF-1.0\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"xref\n0 3\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"trailer<</Size 3/Root 1 0 R>>\n"
            b"startxref\n109\n%%EOF\n"
        )
        path = _write_pdf(tmp_path, minimal_pdf, name="zero.pdf")
        scanner = PDFSecurityScanner(PDFProcessorConfig())
        _, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_PARSE_EMPTY for e in errors)


# ---------------------------------------------------------------------------
# JavaScript Detection Tests
# ---------------------------------------------------------------------------


class TestJavaScript:
    @staticmethod
    def _make_js_pdf(tmp_path) -> str:
        """Create a PDF with embedded JavaScript using PyMuPDF xref API."""
        path = str(tmp_path / "js.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(path)
        doc.close()

        # Re-open and inject JS via xref manipulation
        doc = fitz.open(path)
        js_code = "app.alert('test');"
        xref = doc.get_new_xref()
        doc.update_object(xref, f"<< /Type /Action /S /JavaScript /JS ({js_code}) >>")
        catalog_xref = doc.pdf_catalog()
        doc.xref_set_key(catalog_xref, "OpenAction", f"{xref} 0 R")
        doc.saveIncr()
        doc.close()
        return path

    def test_javascript_detected_and_rejected(self, tmp_path):
        path = self._make_js_pdf(tmp_path)
        scanner = PDFSecurityScanner(PDFProcessorConfig())
        _, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_SECURITY_JAVASCRIPT for e in errors)

    def test_javascript_with_override(self, tmp_path):
        path = self._make_js_pdf(tmp_path)
        config = PDFProcessorConfig(
            reject_javascript=False,
            reject_javascript_override_reason="TICKET-300: Legacy HR forms",
        )
        scanner = PDFSecurityScanner(config)
        _, errors = scanner.scan(path)
        fatal = [e for e in errors if e.code == ErrorCode.E_SECURITY_JAVASCRIPT]
        assert len(fatal) == 0
        overrides = [e for e in errors if e.code == ErrorCode.W_SECURITY_OVERRIDE]
        assert len(overrides) >= 1
        assert "TICKET-300" in overrides[0].message


# ---------------------------------------------------------------------------
# Encryption Tests
# ---------------------------------------------------------------------------


class TestEncryption:
    @staticmethod
    def _make_encrypted_pdf(tmp_path, user_password: str, owner_password: str) -> str:
        """Create an encrypted PDF using PyMuPDF."""
        path = str(tmp_path / "encrypted.pdf")
        doc = fitz.open()
        doc.new_page()
        page = doc[0]
        tw = fitz.TextWriter(page.rect)
        tw.append((100, 700), "Encrypted content")
        tw.write_text(page)
        perm = fitz.PDF_PERM_ACCESSIBILITY | fitz.PDF_PERM_PRINT
        encrypt_meth = fitz.PDF_ENCRYPT_AES_256
        doc.save(
            path,
            encryption=encrypt_meth,
            user_pw=user_password,
            owner_pw=owner_password,
            permissions=perm,
        )
        doc.close()
        return path

    def test_owner_password_only(self, tmp_path):
        """PDF with owner password but empty user password should be accessible."""
        path = self._make_encrypted_pdf(tmp_path, user_password="", owner_password="secret")
        scanner = PDFSecurityScanner(PDFProcessorConfig())
        metadata, errors = scanner.scan(path)
        # Should authenticate with empty password
        fatal = [e for e in errors if e.code == ErrorCode.E_PARSE_PASSWORD]
        assert len(fatal) == 0
        assert metadata.is_encrypted is True

    def test_user_password_required(self, tmp_path):
        """PDF requiring user password should fail."""
        path = self._make_encrypted_pdf(tmp_path, user_password="pass123", owner_password="owner456")
        scanner = PDFSecurityScanner(PDFProcessorConfig())
        metadata, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.E_PARSE_PASSWORD for e in errors)
        assert metadata.needs_password is True


# ---------------------------------------------------------------------------
# Embedded Files Tests
# ---------------------------------------------------------------------------


class TestEmbeddedFiles:
    def test_embedded_files_warning(self, tmp_path):
        """PDF with embedded files should produce W_EMBEDDED_FILES."""
        path = str(tmp_path / "embed.pdf")
        doc = fitz.open()
        doc.new_page()
        # Embed a file
        doc.embfile_add("test.txt", b"embedded content", filename="test.txt")
        doc.save(path)
        doc.close()

        scanner = PDFSecurityScanner(PDFProcessorConfig())
        _, errors = scanner.scan(path)
        assert any(e.code == ErrorCode.W_EMBEDDED_FILES for e in errors)


# ---------------------------------------------------------------------------
# Audit Logging Tests
# ---------------------------------------------------------------------------


class TestAuditLogging:
    def test_override_logged(self, tmp_path, caplog):
        """Security overrides should emit WARNING log entries."""
        config = PDFProcessorConfig(
            max_file_size_mb=0,
            max_file_size_override_reason="TICKET-400: test",
        )
        scanner = PDFSecurityScanner(config)
        path = _write_pdf(tmp_path, _make_simple_pdf())
        with caplog.at_level("WARNING", logger="ingestkit_pdf"):
            scanner.scan(path)
        assert any("SECURITY_OVERRIDE" in r.message for r in caplog.records)
        assert any("TICKET-400" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nonexistent_file(self, scanner):
        """Non-existent file path should return E_SECURITY_INVALID_PDF."""
        _, errors = scanner.scan("/nonexistent/file.pdf")
        assert any(
            e.code in (ErrorCode.E_SECURITY_INVALID_PDF, ErrorCode.E_PARSE_CORRUPT)
            for e in errors
        )

    def test_errors_have_stage(self, scanner, tmp_path):
        """All errors should have stage='security'."""
        path = _write_pdf(tmp_path, b"NOT PDF", name="bad.pdf")
        _, errors = scanner.scan(path)
        for e in errors:
            assert e.stage == "security"

    def test_warnings_are_recoverable(self, scanner, tmp_path):
        """All W_* warnings should have recoverable=True."""
        # Create PDF with embedded files to trigger a warning
        path = str(tmp_path / "warn.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.embfile_add("test.txt", b"data", filename="test.txt")
        doc.save(path)
        doc.close()

        _, errors = scanner.scan(path)
        for e in errors:
            if e.code.value.startswith("W_"):
                assert e.recoverable is True, f"{e.code} should be recoverable"
