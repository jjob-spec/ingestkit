"""Shared fixtures for ingestkit-email tests."""

from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# EML fixtures
# ---------------------------------------------------------------------------

PLAIN_BODY = "Hello, this is a test email body."
HTML_BODY = "<html><body><p>Hello, this is <b>HTML</b> body.</p></body></html>"


def _build_eml_bytes(
    *,
    plain: str | None = PLAIN_BODY,
    html: str | None = None,
    attachment: bool = False,
) -> bytes:
    """Build a minimal valid EML as bytes."""
    if plain and html:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))
    elif html:
        msg = MIMEMultipart()
        msg.attach(MIMEText(html, "html"))
    elif plain:
        msg = MIMEText(plain, "plain")
    else:
        msg = MIMEText("", "plain")

    # Wrap in multipart/mixed if attachment needed
    if attachment:
        outer = MIMEMultipart("mixed")
        outer["From"] = msg.get("From", "sender@example.com")
        outer["To"] = msg.get("To", "recipient@example.com")
        outer["Date"] = msg.get("Date", "Mon, 17 Feb 2026 12:00:00 +0000")
        outer["Subject"] = msg.get("Subject", "Test Subject")
        outer["Message-ID"] = msg.get("Message-ID", "<test-123@example.com>")

        # Re-attach body part(s)
        if isinstance(msg, MIMEMultipart):
            for part in msg.get_payload():
                outer.attach(part)
        else:
            outer.attach(msg)

        # Add binary attachment
        att = MIMEBase("application", "octet-stream")
        att.set_payload(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        encoders.encode_base64(att)
        att.add_header("Content-Disposition", "attachment", filename="image.png")
        outer.attach(att)
        msg = outer

    if "From" not in msg:
        msg["From"] = "sender@example.com"
    if "To" not in msg:
        msg["To"] = "recipient@example.com"
    if "Date" not in msg:
        msg["Date"] = "Mon, 17 Feb 2026 12:00:00 +0000"
    if "Subject" not in msg:
        msg["Subject"] = "Test Subject"
    if "Message-ID" not in msg:
        msg["Message-ID"] = "<test-123@example.com>"

    return msg.as_bytes()


@pytest.fixture
def sample_eml_bytes() -> bytes:
    """Minimal valid multipart EML with plain text body."""
    return _build_eml_bytes(plain=PLAIN_BODY)


@pytest.fixture
def sample_eml_html_only() -> bytes:
    """EML with only text/html body part."""
    return _build_eml_bytes(plain=None, html=HTML_BODY)


@pytest.fixture
def sample_eml_multipart() -> bytes:
    """EML with plain + HTML + binary attachment."""
    return _build_eml_bytes(plain=PLAIN_BODY, html=HTML_BODY, attachment=True)


@pytest.fixture
def sample_eml_file(tmp_path: Path, sample_eml_bytes: bytes) -> str:
    """Write plain-text EML to temp file, return path."""
    p = tmp_path / "test.eml"
    p.write_bytes(sample_eml_bytes)
    return str(p)


@pytest.fixture
def sample_eml_html_file(tmp_path: Path, sample_eml_html_only: bytes) -> str:
    """Write HTML-only EML to temp file, return path."""
    p = tmp_path / "test_html.eml"
    p.write_bytes(sample_eml_html_only)
    return str(p)


@pytest.fixture
def sample_eml_multipart_file(tmp_path: Path, sample_eml_multipart: bytes) -> str:
    """Write multipart EML with attachment to temp file, return path."""
    p = tmp_path / "test_multi.eml"
    p.write_bytes(sample_eml_multipart)
    return str(p)


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock satisfying VectorStoreBackend protocol."""
    store = MagicMock()
    store.upsert_chunks.return_value = 1
    store.ensure_collection.return_value = None
    store.create_payload_index.return_value = None
    store.delete_by_ids.return_value = 0
    return store


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock satisfying EmbeddingBackend protocol (768-dim zero vectors)."""
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 768]
    embedder.dimension.return_value = 768
    return embedder
