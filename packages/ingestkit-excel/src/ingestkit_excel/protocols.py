"""Backend protocols for the ingestkit-excel pipeline.

Defines the four structural-subtyping interfaces that concrete backends must
satisfy.  All protocols are ``@runtime_checkable`` so callers can optionally
verify conformance with ``isinstance`` checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from ingestkit_excel.models import ChunkPayload


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Interface for vector-store backends (e.g. Qdrant, Milvus)."""

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Upsert chunk payloads into the given collection. Returns count upserted."""
        ...

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Create the collection if it does not already exist."""
        ...

    def create_payload_index(self, collection: str, field: str, field_type: str) -> None:
        """Create a payload index on the specified field."""
        ...

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete points by their IDs. Returns count deleted."""
        ...


@runtime_checkable
class StructuredDBBackend(Protocol):
    """Interface for structured-database backends (e.g. SQLite, PostgreSQL)."""

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Write a DataFrame as a table, replacing if it already exists."""
        ...

    def drop_table(self, table_name: str) -> None:
        """Drop a table by name."""
        ...

    def table_exists(self, table_name: str) -> bool:
        """Return True if the table exists in the database."""
        ...

    def get_table_schema(self, table_name: str) -> dict:
        """Return the table schema as ``{column_name: type_string}``."""
        ...

    def get_connection_uri(self) -> str:
        """Return the database connection URI."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Interface for LLM backends (e.g. Ollama)."""

    def classify(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout: float | None = None,
    ) -> dict:
        """Send a classification prompt and return the parsed JSON response."""
        ...

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> str:
        """Send a generation prompt and return the raw text response."""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface for embedding backends (e.g. Ollama nomic-embed-text)."""

    def embed(
        self, texts: list[str], timeout: float | None = None
    ) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations."""
        ...

    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
