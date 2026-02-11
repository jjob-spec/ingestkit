"""Shared test fixtures for ingestkit-excel tests."""

from __future__ import annotations

import pytest

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.models import IngestKey


@pytest.fixture()
def sample_config() -> ExcelProcessorConfig:
    """Return an ExcelProcessorConfig with all defaults."""
    return ExcelProcessorConfig()


@pytest.fixture()
def sample_ingest_key() -> IngestKey:
    """Return a sample IngestKey instance for testing."""
    return IngestKey(
        content_hash="abc123def456",
        source_uri="file:///tmp/test.xlsx",
        parser_version="ingestkit_excel:1.0.0",
        tenant_id="test_tenant",
    )
