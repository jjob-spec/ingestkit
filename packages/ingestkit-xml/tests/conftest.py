"""Shared test fixtures for ingestkit-xml tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestkit_xml.config import XMLProcessorConfig


@pytest.fixture
def default_config() -> XMLProcessorConfig:
    """Return a default XMLProcessorConfig."""
    return XMLProcessorConfig()


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
def tmp_xml_file(tmp_path: Path):
    """Factory fixture to write XML string to a temp .xml file and return the path."""

    def _write(xml_content: str, filename: str = "test.xml") -> str:
        file_path = tmp_path / filename
        file_path.write_text(xml_content, encoding="utf-8")
        return str(file_path)

    return _write


@pytest.fixture
def sample_xml_simple() -> str:
    """Simple XML string with a few elements."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <title>Test Document</title>
    <body>Hello world</body>
</root>"""


@pytest.fixture
def sample_xml_nested() -> str:
    """Deeply nested XML."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <level1>
        <level2>
            <level3>
                <level4>Deep content</level4>
            </level3>
        </level2>
    </level1>
</root>"""


@pytest.fixture
def sample_xml_namespaced() -> str:
    """XML with namespace declarations."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<root xmlns:ns="http://example.com/ns" xmlns:other="http://example.com/other">
    <ns:item>Namespaced item</ns:item>
    <other:data>Other data</other:data>
</root>"""


@pytest.fixture
def sample_xml_with_attrs() -> str:
    """XML with meaningful attributes."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <item id="1" type="widget">First item</item>
    <item id="2" type="gadget" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">Second item</item>
</root>"""
