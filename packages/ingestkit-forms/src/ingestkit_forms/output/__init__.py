"""Output writers for extracted form data.

Subpackage containing:
- db_writer: Structured DB row writer with schema evolution
- chunk_writer: RAG chunk serializer + embedder
- dual_writer: Orchestrator with consistency modes and rollback
"""

from ingestkit_forms.output.chunk_writer import FormChunkWriter
from ingestkit_forms.output.db_writer import FormDBWriter
from ingestkit_forms.output.dual_writer import FormDualWriter, rollback_written_artifacts

__all__ = [
    "FormDBWriter",
    "FormChunkWriter",
    "FormDualWriter",
    "rollback_written_artifacts",
]
