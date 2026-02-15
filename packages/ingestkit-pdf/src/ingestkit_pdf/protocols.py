"""Backend protocols for the ingestkit-pdf pipeline.

Re-exports the four structural-subtyping interfaces from ``ingestkit_core``
plus the PDF-specific ``ExecutionBackend`` from ``ingestkit_pdf.execution``.
All existing import paths (``from ingestkit_pdf.protocols import ...``)
continue to work.
"""

from ingestkit_core.protocols import (
    EmbeddingBackend,
    LLMBackend,
    StructuredDBBackend,
    VectorStoreBackend,
)
from ingestkit_pdf.execution import ExecutionBackend

__all__ = [
    "VectorStoreBackend",
    "StructuredDBBackend",
    "LLMBackend",
    "EmbeddingBackend",
    "ExecutionBackend",
]
