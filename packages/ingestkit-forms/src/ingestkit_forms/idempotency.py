"""Idempotency key computation for form ingestion.

Three-tier keying system (spec section 4.3):

1. **Global ingest key** -- delegates to ``ingestkit_core`` for document-level
   deduplication based on file content, parser version, and optional tenant ID.
2. **Form extraction key** -- ``sha256(ingest_key_global | template_id |
   template_version)`` produces a 64-char hex digest that changes whenever the
   template is revised, enabling safe re-extraction without collisions.
3. **Vector point ID** -- ``uuid5(NAMESPACE_URL, form_extraction_key :
   chunk_index)`` yields deterministic UUIDs for each chunk, suitable for
   upsert into a vector store.
"""

from __future__ import annotations

import hashlib
import uuid

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import IngestKey

__all__ = [
    "IngestKey",
    "compute_ingest_key",
    "compute_form_extraction_key",
    "compute_vector_point_id",
]


def compute_form_extraction_key(
    ingest_key_global: str,
    template_id: str,
    template_version: int,
) -> str:
    """Compute a template-versioned extraction key.

    Parameters
    ----------
    ingest_key_global:
        The 64-char hex global ingest key (``IngestKey.key``).
    template_id:
        Unique identifier for the form template.
    template_version:
        Integer version of the template.  A new version produces a new
        extraction key even for the same document.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest.
    """
    payload = f"{ingest_key_global}|{template_id}|{template_version}"
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_vector_point_id(
    form_extraction_key: str,
    chunk_index: int,
) -> str:
    """Compute a deterministic UUID for a vector-store point.

    Parameters
    ----------
    form_extraction_key:
        The 64-char hex form extraction key.
    chunk_index:
        Zero-based index of the chunk within the extraction.

    Returns
    -------
    str
        UUID string (``uuid5(NAMESPACE_URL, key:index)``).
    """
    name = f"{form_extraction_key}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))
