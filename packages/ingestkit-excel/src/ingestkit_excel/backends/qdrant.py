"""Qdrant backend for the VectorStoreBackend protocol.

Provides a concrete implementation backed by ``qdrant-client``.  The client
library is an optional dependency -- importing this module when ``qdrant-client``
is not installed will raise ``ImportError`` at class instantiation time only.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestkit_core.models import ChunkPayload
    from ingestkit_excel.config import ExcelProcessorConfig

logger = logging.getLogger("ingestkit_excel")


class QdrantVectorStore:
    """Qdrant-backed vector store.

    Satisfies :class:`~ingestkit_core.protocols.VectorStoreBackend` via
    structural subtyping (no inheritance required).

    Parameters
    ----------
    url:
        Qdrant server URL (e.g. ``"http://localhost:6333"``).
    collection_prefix:
        Prefix prepended to collection names for multi-tenant isolation.
    config:
        Pipeline configuration providing timeout and retry settings.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_prefix: str = "",
        config: ExcelProcessorConfig | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install 'ingestkit-excel[qdrant]'"
            ) from exc

        from ingestkit_excel.config import ExcelProcessorConfig

        self._config = config or ExcelProcessorConfig()
        self._prefix = collection_prefix
        self._client = QdrantClient(
            url=url,
            timeout=self._config.backend_timeout_seconds,
        )

    def _prefixed(self, collection: str) -> str:
        """Return the collection name with the configured prefix."""
        if self._prefix:
            return f"{self._prefix}_{collection}"
        return collection

    def _retry(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        """Execute *fn* with exponential-backoff retries.

        Uses ``config.backend_max_retries`` and ``config.backend_backoff_base``.
        """
        last_exc: Exception | None = None
        max_attempts = 1 + self._config.backend_max_retries
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "Qdrant operation failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_attempts,
                        sleep_time,
                        exc,
                    )
                    time.sleep(sleep_time)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Create the collection if it does not already exist.

        Uses cosine distance metric.
        """
        from qdrant_client.models import Distance, VectorParams

        name = self._prefixed(collection)

        def _create() -> None:
            if not self._client.collection_exists(name):
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", name, vector_size)

        try:
            self._retry(_create)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to ensure Qdrant vector collection '{name}': {exc}"
            ) from exc

    def upsert_chunks(self, collection: str, chunks: list[ChunkPayload]) -> int:
        """Upsert chunk payloads into the given collection.

        Converts each ``ChunkPayload`` to a Qdrant ``PointStruct`` and performs
        a batch upsert with retry.

        Returns the number of points upserted.
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0

        name = self._prefixed(collection)
        points = [
            PointStruct(
                id=chunk.id,
                vector=chunk.vector,
                payload={
                    "text": chunk.text,
                    **chunk.metadata.model_dump(),
                },
            )
            for chunk in chunks
        ]

        def _upsert() -> None:
            self._client.upsert(collection_name=name, points=points)

        try:
            self._retry(_upsert)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to upsert {len(chunks)} chunks to Qdrant vector collection '{name}': {exc}"
            ) from exc

        logger.info("Upserted %d points to Qdrant collection '%s'", len(points), name)
        return len(points)

    def create_payload_index(
        self, collection: str, field: str, field_type: str
    ) -> None:
        """Create a payload index on the specified field.

        Parameters
        ----------
        field_type:
            One of ``"keyword"`` or ``"integer"``.
        """
        from qdrant_client.models import PayloadSchemaType

        type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
        }
        schema_type = type_map.get(field_type)
        if schema_type is None:
            raise ValueError(
                f"Unsupported field_type '{field_type}'. Use 'keyword' or 'integer'."
            )

        name = self._prefixed(collection)

        def _create_index() -> None:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema_type,
            )

        try:
            self._retry(_create_index)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to create payload index on Qdrant vector collection '{name}.{field}': {exc}"
            ) from exc

    def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete points by their IDs. Returns count deleted."""
        from qdrant_client.models import PointIdsList

        if not ids:
            return 0

        name = self._prefixed(collection)

        def _delete() -> None:
            self._client.delete(
                collection_name=name,
                points_selector=PointIdsList(points=ids),
            )

        try:
            self._retry(_delete)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to delete {len(ids)} points from Qdrant vector collection '{name}': {exc}"
            ) from exc

        logger.info("Deleted %d points from Qdrant collection '%s'", len(ids), name)
        return len(ids)
