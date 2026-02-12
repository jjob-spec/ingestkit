"""Deterministic ingest-key computation for deduplication.

This module provides :func:`compute_ingest_key`, which produces an
:class:`~ingestkit_core.models.IngestKey` from a file on disk.  The
resulting key is fully deterministic: identical file content, parser
version, and tenant ID will always yield the same
:pyattr:`IngestKey.key` digest.

The package **provides** the key but does **not** enforce any
deduplication policy -- that responsibility belongs to the caller.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ingestkit_core.models import IngestKey


def compute_ingest_key(
    file_path: str,
    parser_version: str,
    tenant_id: str | None = None,
    source_uri: str | None = None,
) -> IngestKey:
    """Compute a deterministic ingest key for deduplication.

    Reads the raw bytes of *file_path*, computes a SHA-256 content hash,
    and combines it with the parser version and optional identifiers to
    produce an :class:`IngestKey`.

    Parameters
    ----------
    file_path:
        Path to the file to hash.
    parser_version:
        Parser version string (e.g. ``"ingestkit_excel:1.0.0"``).
    tenant_id:
        Optional tenant identifier for multi-tenant scenarios.
    source_uri:
        Optional override for the source URI stored in the key.  When
        *None*, the canonical absolute POSIX path of *file_path* is used.

    Returns
    -------
    IngestKey
        A populated :class:`IngestKey` whose :pyattr:`~IngestKey.key`
        property yields the composite SHA-256 hex digest.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    OSError
        If the file cannot be read.
    """
    # 1. Compute content hash from raw file bytes.
    file_bytes = Path(file_path).read_bytes()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # 2. Derive source URI -- canonical absolute path or caller override.
    if source_uri is None:
        source_uri = Path(file_path).resolve().as_posix()

    # 3. Construct and return the IngestKey.
    return IngestKey(
        content_hash=content_hash,
        source_uri=source_uri,
        parser_version=parser_version,
        tenant_id=tenant_id,
    )
