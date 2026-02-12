"""Processing-path implementations for ingestkit-excel."""

from ingestkit_excel.processors.serializer import TextSerializer
from ingestkit_excel.processors.structured_db import StructuredDBProcessor

__all__ = ["StructuredDBProcessor", "TextSerializer"]
