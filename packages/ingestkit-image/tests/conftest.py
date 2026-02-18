"""Shared fixtures for ingestkit-image tests."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from ingestkit_image.config import ImageProcessorConfig
from ingestkit_image.models import ImageMetadata, ImageType


# ---------------------------------------------------------------------------
# Mock Backend Classes
# ---------------------------------------------------------------------------


class MockVLMBackend:
    """Mock VLM backend satisfying ImageVLMBackend protocol."""

    def __init__(
        self,
        caption_text: str = "A photo of a building with a red roof.",
        available: bool = True,
        raise_on_caption: Exception | None = None,
    ) -> None:
        self._caption_text = caption_text
        self._available = available
        self._raise_on_caption = raise_on_caption
        self.caption_calls: list[dict] = []

    def caption(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        temperature: float = 0.3,
        timeout: float | None = None,
    ) -> str:
        self.caption_calls.append({
            "image_bytes_len": len(image_bytes),
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "timeout": timeout,
        })
        if self._raise_on_caption is not None:
            raise self._raise_on_caption
        return self._caption_text

    def model_name(self) -> str:
        return "mock-vlm:test"

    def is_available(self) -> bool:
        return self._available


class MockVectorStore:
    """Mock VectorStoreBackend."""

    def __init__(self) -> None:
        self.upserted: list = []
        self.collections_ensured: list[str] = []

    def upsert_chunks(self, collection: str, chunks: list) -> int:
        self.upserted.extend(chunks)
        return len(chunks)

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        self.collections_ensured.append(collection)

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        pass

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        return 0


class MockEmbedder:
    """Mock EmbeddingBackend returning fixed-dimension vectors."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_config() -> ImageProcessorConfig:
    """Return an ImageProcessorConfig with defaults."""
    return ImageProcessorConfig()


@pytest.fixture
def mock_vlm_backend() -> MockVLMBackend:
    """Return a mock VLM backend with default caption."""
    return MockVLMBackend()


@pytest.fixture
def mock_vector_store() -> MockVectorStore:
    return MockVectorStore()


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def sample_image_path(tmp_path) -> str:
    """Create a small 100x100 PNG image in a temp directory. Returns path."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = str(tmp_path / "test_image.png")
    img.save(path, format="PNG")
    return path


@pytest.fixture
def sample_jpeg_path(tmp_path) -> str:
    """Create a small 100x100 JPEG image. Returns path."""
    img = Image.new("RGB", (100, 100), color=(0, 128, 255))
    path = str(tmp_path / "test_image.jpg")
    img.save(path, format="JPEG")
    return path


@pytest.fixture
def sample_image_metadata(sample_image_path) -> ImageMetadata:
    """Return ImageMetadata for the sample image."""
    import hashlib
    content_hash = hashlib.sha256(open(sample_image_path, "rb").read()).hexdigest()
    return ImageMetadata(
        file_path=sample_image_path,
        file_size_bytes=os.path.getsize(sample_image_path),
        image_type=ImageType.PNG,
        width=100,
        height=100,
        content_hash=content_hash,
        has_exif=False,
        color_mode="RGB",
    )
