"""Shared test fixtures for ingestkit-json tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ingestkit_json.config import JSONProcessorConfig


@pytest.fixture
def default_config() -> JSONProcessorConfig:
    """Return a default JSONProcessorConfig."""
    return JSONProcessorConfig()


@pytest.fixture
def sample_json_nested() -> dict:
    """Return a nested dict for testing."""
    return {
        "user": {
            "name": "Alice",
            "address": {
                "city": "Springfield",
                "state": "IL",
            },
        },
        "active": True,
        "score": 42.5,
    }


@pytest.fixture
def sample_json_array() -> list:
    """Return a root-level array."""
    return [
        {"id": 1, "name": "Widget"},
        {"id": 2, "name": "Gadget"},
    ]


@pytest.fixture
def sample_json_flat() -> dict:
    """Return a simple flat object."""
    return {"a": 1, "b": "hello", "c": True}


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
def tmp_json_file(tmp_path: Path):
    """Factory fixture to write JSON data to a temp file and return the path."""

    def _write(data: Any, filename: str = "test.json") -> str:
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return str(file_path)

    return _write
