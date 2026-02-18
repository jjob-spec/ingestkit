"""Shared test fixtures for ingestkit-doc tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestkit_doc.config import DocProcessorConfig

# OLE2 magic bytes used by .doc files
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@pytest.fixture
def default_config() -> DocProcessorConfig:
    """Return a default DocProcessorConfig."""
    return DocProcessorConfig()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Return a mock VectorStoreBackend."""
    mock = MagicMock()
    mock.upsert_chunks.return_value = 1
    mock.ensure_collection.return_value = None
    return mock


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Return a mock EmbeddingBackend returning deterministic vectors."""
    mock = MagicMock()
    mock.embed.return_value = [[0.1] * 768]
    mock.dimension.return_value = 768
    return mock


@pytest.fixture
def tmp_doc_file(tmp_path: Path):
    """Factory fixture to write binary content to a .doc temp file and return the path."""

    def _write(content: bytes | None = None, filename: str = "test.doc") -> str:
        file_path = tmp_path / filename
        if content is None:
            # Write OLE2 magic header + some padding (not a valid Word file,
            # but passes security scan magic byte check)
            content = _OLE2_MAGIC + b"\x00" * 504
        file_path.write_bytes(content)
        return str(file_path)

    return _write
