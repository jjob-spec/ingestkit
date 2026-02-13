"""SQLite backend for the StructuredDBBackend protocol.

Provides a concrete implementation backed by Python's built-in ``sqlite3``
module.  Suitable for local / single-node deployments and testing.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("ingestkit_excel")


class SQLiteStructuredDB:
    """SQLite-backed structured database.

    Satisfies :class:`~ingestkit_core.protocols.StructuredDBBackend` via
    structural subtyping (no inheritance required).

    Parameters
    ----------
    db_path:
        Filesystem path or ``":memory:"`` for an in-memory database.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        try:
            self._conn = sqlite3.connect(db_path)
        except sqlite3.Error as exc:
            raise ConnectionError(
                f"Failed to connect to SQLite database at {db_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def create_table_from_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Write a DataFrame as a table, replacing if it already exists.

        Uses ``pandas.DataFrame.to_sql`` with ``if_exists='replace'``.

        Raises
        ------
        RuntimeError
            If the write fails.
        """
        import pandas as pd  # noqa: F811 -- runtime import

        try:
            df.to_sql(table_name, self._conn, if_exists="replace", index=False)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            raise RuntimeError(
                f"Failed to write table '{table_name}': {exc}"
            ) from exc

    def drop_table(self, table_name: str) -> None:
        """Drop a table by name (no-op if it does not exist)."""
        try:
            self._conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Failed to drop table '{table_name}': {exc}"
            ) from exc

    def table_exists(self, table_name: str) -> bool:
        """Return True if the table exists in the database."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone()[0] > 0

    def get_table_schema(self, table_name: str) -> dict:
        """Return the table schema as ``{column_name: type_string}``.

        Uses ``PRAGMA table_info`` to inspect column definitions.
        """
        cursor = self._conn.execute(f"PRAGMA table_info([{table_name}])")
        rows = cursor.fetchall()
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        return {row[1]: row[2] for row in rows}

    def get_connection_uri(self) -> str:
        """Return the database connection URI."""
        return f"sqlite:///{self._db_path}"

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
