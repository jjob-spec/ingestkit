"""Milvus backend stub for the VectorStoreBackend protocol.

This module provides a placeholder implementation that raises
``NotImplementedError`` for all methods.  It exists so that the backend
registry can list Milvus as a known option and provide a clear error
message when a caller attempts to use it before a real implementation
is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestkit_core.models import ChunkPayload


class MilvusVectorStore:
    """Milvus vector store stub.

    All methods raise ``NotImplementedError``.  Install and configure a
    real Milvus backend when ready for production use.
    """

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Not implemented."""
        raise NotImplementedError("MilvusVectorStore is a stub — not yet implemented.")

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Not implemented."""
        raise NotImplementedError("MilvusVectorStore is a stub — not yet implemented.")

    def create_payload_index(
        self, collection: str, field: str, field_type: str
    ) -> None:
        """Not implemented."""
        raise NotImplementedError("MilvusVectorStore is a stub — not yet implemented.")

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Not implemented."""
        raise NotImplementedError("MilvusVectorStore is a stub — not yet implemented.")
