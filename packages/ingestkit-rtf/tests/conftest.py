"""Shared test fixtures for ingestkit-rtf tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestkit_rtf.config import RTFProcessorConfig

# RTF magic bytes
_RTF_MAGIC = b"{\\rtf"


@pytest.fixture
def default_config() -> RTFProcessorConfig:
    """Return a default RTFProcessorConfig."""
    return RTFProcessorConfig()


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
def tmp_rtf_file(tmp_path: Path):
    """Factory fixture to write content to an .rtf temp file and return the path."""

    def _write(content: bytes | None = None, filename: str = "test.rtf") -> str:
        file_path = tmp_path / filename
        if content is None:
            # Write a minimal valid RTF file
            content = b"{\\rtf1\\ansi Hello world.}"
        file_path.write_bytes(content)
        return str(file_path)

    return _write
