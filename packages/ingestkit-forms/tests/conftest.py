"""Shared test fixtures for ingestkit-forms tests.

Provides mock backends, test configuration, and fixture placeholders
for form template, PDF, Excel, and image test files.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def form_config():
    """Return a FormProcessorConfig with all defaults.

    TODO: Replace with actual FormProcessorConfig once config.py is implemented.
    """
    return {}


# ---------------------------------------------------------------------------
# Mock Backend Fixtures (Placeholders)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_template_store():
    """Mock FormTemplateStore for testing.

    TODO: Implement MockFormTemplateStore once protocols.py is implemented.
    """
    return None


@pytest.fixture()
def mock_ocr_backend():
    """Mock OCRBackend for testing.

    TODO: Implement MockOCRBackend once protocols.py is implemented.
    """
    return None


@pytest.fixture()
def mock_pdf_widget_backend():
    """Mock PDFWidgetBackend for testing.

    TODO: Implement MockPDFWidgetBackend once protocols.py is implemented.
    """
    return None


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
