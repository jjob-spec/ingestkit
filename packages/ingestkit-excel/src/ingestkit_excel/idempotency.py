"""Deterministic ingest-key computation for deduplication.

Re-exports ``compute_ingest_key`` and ``IngestKey`` from ``ingestkit_core``.
All existing import paths continue to work.
"""

from ingestkit_core.idempotency import compute_ingest_key
from ingestkit_core.models import IngestKey

__all__ = [
    "compute_ingest_key",
    "IngestKey",
]
