"""Concrete backend implementations for ingestkit-excel.

Exports all six backend classes.  The concrete backends for Qdrant and Ollama
require optional dependencies (``qdrant-client`` and ``httpx`` respectively).
They are guarded with lazy ``try/except ImportError`` blocks so the package
works even when optional dependencies are not installed.

Stubs (Milvus, PostgreSQL) are always importable and raise
``NotImplementedError`` on all methods.
"""

from __future__ import annotations

# --- Always-available backends (stdlib only) ---
from ingestkit_excel.backends.sqlite import SQLiteStructuredDB

# --- Stubs (no external deps) ---
from ingestkit_excel.backends.milvus import MilvusVectorStore
from ingestkit_excel.backends.postgres import PostgresStructuredDB

# --- Optional-dep backends (lazy import guards) ---
QdrantVectorStore = None
OllamaLLM = None
OllamaEmbedding = None

try:
    from ingestkit_excel.backends.qdrant import QdrantVectorStore  # type: ignore[assignment]
except ImportError:
    pass

try:
    from ingestkit_excel.backends.ollama import OllamaLLM, OllamaEmbedding  # type: ignore[assignment]
except ImportError:
    pass

__all__ = [
    "SQLiteStructuredDB",
    "QdrantVectorStore",
    "OllamaLLM",
    "OllamaEmbedding",
    "MilvusVectorStore",
    "PostgresStructuredDB",
]
