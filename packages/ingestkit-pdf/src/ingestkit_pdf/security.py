"""Pre-flight security scanner for PDF files.

Rejects dangerous or oversized PDFs before any extraction begins.
Implements the checks from SPEC §7.1-§7.5 including magic byte validation,
size/page limits, JavaScript detection, decompression bomb detection,
encryption handling, and security override governance with audit logging.
"""

from __future__ import annotations

import logging
import os

import fitz  # PyMuPDF

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import DocumentMetadata

logger = logging.getLogger("ingestkit_pdf")

_PDF_MAGIC = b"%PDF-"


class PDFSecurityScanner:
    """Run pre-flight security checks on a PDF file.

    Returns document metadata and a list of errors/warnings. Fatal errors
    (``E_*`` codes) mean the file should not be processed further.
    """

    def __init__(self, config: PDFProcessorConfig) -> None:
        self.config = config

    def scan(self, file_path: str) -> tuple[DocumentMetadata, list[IngestError]]:
        """Run all pre-flight checks.

        Returns:
            A tuple of (DocumentMetadata, list of errors/warnings).
            Fatal errors have codes starting with ``E_``.
        """
        errors: list[IngestError] = []
        metadata = DocumentMetadata()

        # --- 1. Magic bytes ---
        if not self._check_magic_bytes(file_path):
            errors.append(
                IngestError(
                    code=ErrorCode.E_SECURITY_INVALID_PDF,
                    message=f"File does not start with %PDF- magic bytes: {file_path}",
                    stage="security",
                )
            )
            return metadata, errors

        # --- 2. File size ---
        file_size = os.path.getsize(file_path)
        metadata.file_size_bytes = file_size

        max_bytes = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            if self.config.max_file_size_override_reason:
                self._log_override(file_path, "max_file_size_mb", self.config.max_file_size_override_reason)
                errors.append(
                    IngestError(
                        code=ErrorCode.W_SECURITY_OVERRIDE,
                        message=(
                            f"max_file_size_mb override: file is {file_size} bytes "
                            f"(limit {max_bytes}), reason: {self.config.max_file_size_override_reason}"
                        ),
                        stage="security",
                        recoverable=True,
                    )
                )
            else:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_SECURITY_TOO_LARGE,
                        message=f"File size {file_size} bytes exceeds limit of {max_bytes} bytes",
                        stage="security",
                    )
                )
                return metadata, errors

        # --- 3. Open with PyMuPDF ---
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_PARSE_CORRUPT,
                    message=f"PyMuPDF cannot open file: {exc}",
                    stage="security",
                )
            )
            return metadata, errors

        try:
            metadata = self._extract_metadata(doc, file_path, file_size)

            # --- 4. Encryption check ---
            if doc.needs_pass:
                if doc.authenticate(""):
                    metadata.is_encrypted = True
                    metadata.needs_password = False
                else:
                    metadata.is_encrypted = True
                    metadata.needs_password = True
                    errors.append(
                        IngestError(
                            code=ErrorCode.E_PARSE_PASSWORD,
                            message="PDF requires a password to open",
                            stage="security",
                        )
                    )
                    return metadata, errors

            # --- 5. Page count ---
            if doc.page_count > self.config.max_page_count:
                if self.config.max_page_count_override_reason:
                    self._log_override(file_path, "max_page_count", self.config.max_page_count_override_reason)
                    errors.append(
                        IngestError(
                            code=ErrorCode.W_SECURITY_OVERRIDE,
                            message=(
                                f"max_page_count override: {doc.page_count} pages "
                                f"(limit {self.config.max_page_count}), "
                                f"reason: {self.config.max_page_count_override_reason}"
                            ),
                            stage="security",
                            recoverable=True,
                        )
                    )
                else:
                    errors.append(
                        IngestError(
                            code=ErrorCode.E_SECURITY_TOO_MANY_PAGES,
                            message=(
                                f"Page count {doc.page_count} exceeds limit "
                                f"of {self.config.max_page_count}"
                            ),
                            stage="security",
                        )
                    )
                    return metadata, errors

            # --- 6. Zero pages ---
            if doc.page_count == 0:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_PARSE_EMPTY,
                        message="PDF has zero pages",
                        stage="security",
                    )
                )
                return metadata, errors

            # --- 7. Decompression bomb detection ---
            xref_count = doc.xref_length()
            if xref_count > self.config.max_decompression_ratio * 1000:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_SECURITY_DECOMPRESSION_BOMB,
                        message=(
                            f"Excessive xref objects ({xref_count}), "
                            f"possible decompression bomb"
                        ),
                        stage="security",
                    )
                )
                return metadata, errors

            # --- 8. JavaScript detection ---
            has_js = self._detect_javascript(doc)
            if has_js:
                if self.config.reject_javascript:
                    errors.append(
                        IngestError(
                            code=ErrorCode.E_SECURITY_JAVASCRIPT,
                            message="Embedded JavaScript detected in PDF",
                            stage="security",
                        )
                    )
                    return metadata, errors
                elif self.config.reject_javascript_override_reason:
                    self._log_override(file_path, "reject_javascript", self.config.reject_javascript_override_reason)
                    errors.append(
                        IngestError(
                            code=ErrorCode.W_SECURITY_OVERRIDE,
                            message=(
                                f"reject_javascript override: JS detected, "
                                f"reason: {self.config.reject_javascript_override_reason}"
                            ),
                            stage="security",
                            recoverable=True,
                        )
                    )

            # --- 9. Signed PDF detection ---
            if metadata.is_signed:
                errors.append(
                    IngestError(
                        code=ErrorCode.W_DOCUMENT_SIGNED,
                        message="PDF is digitally signed, read-only extraction",
                        stage="security",
                        recoverable=True,
                    )
                )

            # --- 10. Embedded files detection ---
            if doc.embfile_count() > 0:
                errors.append(
                    IngestError(
                        code=ErrorCode.W_EMBEDDED_FILES,
                        message=f"PDF contains {doc.embfile_count()} embedded file(s), not processed",
                        stage="security",
                        recoverable=True,
                    )
                )

        finally:
            doc.close()

        return metadata, errors

    @staticmethod
    def _check_magic_bytes(file_path: str) -> bool:
        """Check if the file starts with %PDF- magic bytes."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(len(_PDF_MAGIC))
            return header == _PDF_MAGIC
        except OSError:
            return False

    def _extract_metadata(
        self, doc: fitz.Document, file_path: str, file_size: int
    ) -> DocumentMetadata:
        """Extract document metadata from a PyMuPDF document."""
        meta = doc.metadata or {}

        is_signed = self._detect_signatures(doc)
        encryption = meta.get("encryption")
        is_encrypted = bool(encryption) and encryption != "none"

        return DocumentMetadata(
            title=meta.get("title") or None,
            author=meta.get("author") or None,
            subject=meta.get("subject") or None,
            keywords=meta.get("keywords") or None,
            creator=meta.get("creator") or None,
            producer=meta.get("producer") or None,
            creation_date=meta.get("creationDate") or None,
            modification_date=meta.get("modDate") or None,
            pdf_version=meta.get("format") or None,
            page_count=doc.page_count,
            file_size_bytes=file_size,
            is_encrypted=is_encrypted,
            needs_password=bool(doc.needs_pass),
            is_signed=is_signed,
            has_form_fields=doc.is_form_pdf,
            is_linearized=bool(doc.is_fast_webaccess),
        )

    @staticmethod
    def _detect_javascript(doc: fitz.Document) -> bool:
        """Scan xref objects for embedded JavaScript."""
        for xref in range(1, doc.xref_length()):
            try:
                keys = doc.xref_get_keys(xref)
                if "JS" in keys or "JavaScript" in keys:
                    return True
                obj_type = doc.xref_get_key(xref, "S")
                if obj_type and obj_type[1] == "/JavaScript":
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _detect_signatures(doc: fitz.Document) -> bool:
        """Detect digital signatures by scanning for /Sig form fields."""
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                for widget in page.widgets() or []:
                    if widget.field_type_string == "Signature":
                        return True
            except Exception:
                continue
        return False

    @staticmethod
    def _log_override(file_path: str, override_name: str, reason: str) -> None:
        """Emit a W_SECURITY_OVERRIDE audit log entry."""
        logger.warning(
            "ingestkit_pdf | SECURITY_OVERRIDE | file=%s | override=%s | reason=%r | config_source=config",
            os.path.basename(file_path),
            override_name,
            reason,
        )
