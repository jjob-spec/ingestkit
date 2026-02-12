"""Backend protocols for the ingestkit-pdf pipeline.

Re-exports the four structural-subtyping interfaces from ``ingestkit_core``.
All existing import paths (``from ingestkit_pdf.protocols import ...``)
continue to work.
"""

from ingestkit_core.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)

__all__ = [
    "VectorStoreBackend",
    "StructuredDBBackend",
    "LLMBackend",
    "EmbeddingBackend",
]
