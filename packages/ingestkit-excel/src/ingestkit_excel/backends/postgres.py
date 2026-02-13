"""PostgreSQL backend stub for the StructuredDBBackend protocol.

This module provides a placeholder implementation that raises
``NotImplementedError`` for all methods.  It exists so that the backend
registry can list PostgreSQL as a known option and provide a clear error
message when a caller attempts to use it before a real implementation
is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class PostgresStructuredDB:
    """PostgreSQL structured database stub.

    All methods raise ``NotImplementedError``.  Install and configure a
    real PostgreSQL backend when ready for production use.
    """

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Not implemented."""
        raise NotImplementedError(
            "PostgresStructuredDB is a stub — not yet implemented."
        )

    def drop_table(self, table_name: str) -> None:
        """Not implemented."""
        raise NotImplementedError(
            "PostgresStructuredDB is a stub — not yet implemented."
        )

    def table_exists(self, table_name: str) -> bool:
        """Not implemented."""
        raise NotImplementedError(
            "PostgresStructuredDB is a stub — not yet implemented."
        )

    def get_table_schema(self, table_name: str) -> dict:
        """Not implemented."""
        raise NotImplementedError(
            "PostgresStructuredDB is a stub — not yet implemented."
        )

    def get_connection_uri(self) -> str:
        """Not implemented."""
        raise NotImplementedError(
            "PostgresStructuredDB is a stub — not yet implemented."
        )
