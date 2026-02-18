"""Structured DB row writer for extracted form data.

Writes validated form field values as structured rows to the
FormDBBackend. Handles table creation, schema evolution, and
upsert with retry logic.

See spec section 8.1 for authoritative definitions.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from ingestkit_forms.config import FormProcessorConfig
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    ExtractedField,
    FieldType,
    FormExtractionResult,
    FormTemplate,
)
from ingestkit_forms.protocols import FormDBBackend

logger = logging.getLogger("ingestkit_forms")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_TYPE_TO_SQL: dict[FieldType, str] = {
    FieldType.TEXT: "TEXT",
    FieldType.NUMBER: "REAL",
    FieldType.DATE: "TEXT",
    FieldType.CHECKBOX: "INTEGER",
    FieldType.RADIO: "TEXT",
    FieldType.SIGNATURE: "INTEGER",
    FieldType.DROPDOWN: "TEXT",
}

METADATA_COLUMNS: dict[str, str] = {
    "_form_id": "TEXT PRIMARY KEY",
    "_template_id": "TEXT",
    "_template_version": "INTEGER",
    "_source_uri": "TEXT",
    "_ingest_key": "TEXT",
    "_ingest_run_id": "TEXT",
    "_tenant_id": "TEXT",
    "_extracted_at": "TEXT",
    "_overall_confidence": "REAL",
    "_extraction_method": "TEXT",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def slugify_template_name(name: str) -> str:
    """Convert a template name to a DB-safe slug.

    Rules:
    1. Lowercase
    2. Replace non-alphanumeric/underscore with underscore
    3. Collapse consecutive underscores
    4. Strip leading/trailing underscores

    Same logic as ``clean_name`` in ingestkit-excel's ``structured_db.py``.
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9_]", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return slug


def get_table_name(config: FormProcessorConfig, template: FormTemplate) -> str:
    """Return the DB table name for a given template.

    Raises FormIngestException if the generated name is not a safe SQL identifier.
    """
    from ingestkit_forms.security import validate_table_name

    table_name = f"{config.form_db_table_prefix}{slugify_template_name(template.name)}"

    error = validate_table_name(table_name)
    if error is not None:
        raise FormIngestException(
            code=FormErrorCode.E_FORM_TEMPLATE_INVALID,
            message=(
                f"Generated table name '{table_name}' is not a safe SQL identifier: "
                f"{error}. Template: '{template.name}', prefix: '{config.form_db_table_prefix}'"
            ),
            stage="output",
            recoverable=False,
        )

    return table_name


def generate_table_schema(template: FormTemplate) -> dict[str, str]:
    """Generate the full column schema for a template's table.

    Returns a dict of ``{column_name: sql_type}`` combining the 10
    metadata columns and one column per template field.
    """
    schema = dict(METADATA_COLUMNS)
    for field in template.fields:
        sql_type = FIELD_TYPE_TO_SQL.get(field.field_type, "TEXT")
        schema[field.field_name] = sql_type
    return schema


def build_row_dict(
    extraction: FormExtractionResult,
    config: FormProcessorConfig,
    ingest_key: str,
    ingest_run_id: str,
) -> dict[str, Any]:
    """Build a row dict from an extraction result.

    Metadata columns are populated from the extraction and config.
    Field values are coerced: CHECKBOX and SIGNATURE booleans become
    integers (True->1, False->0, None->None).
    """
    row: dict[str, Any] = {
        "_form_id": extraction.form_id,
        "_template_id": extraction.template_id,
        "_template_version": extraction.template_version,
        "_source_uri": extraction.source_uri,
        "_ingest_key": ingest_key,
        "_ingest_run_id": ingest_run_id,
        "_tenant_id": config.tenant_id,
        "_extracted_at": datetime.now(timezone.utc).isoformat(),
        "_overall_confidence": extraction.overall_confidence,
        "_extraction_method": extraction.extraction_method,
    }

    for field in extraction.fields:
        row[field.field_name] = _coerce_field_value(field)

    return row


def _coerce_field_value(field: ExtractedField) -> Any:
    """Coerce a field value for DB storage.

    CHECKBOX and SIGNATURE fields are stored as INTEGER (1/0/None).
    All other types pass through unchanged.
    """
    if field.field_type in (FieldType.CHECKBOX, FieldType.SIGNATURE):
        if field.value is None:
            return None
        return 1 if field.value else 0
    return field.value


# ---------------------------------------------------------------------------
# FormDBWriter
# ---------------------------------------------------------------------------


class FormDBWriter:
    """Writes extracted form data as structured DB rows.

    Handles table creation, schema evolution (adding new columns for
    new template fields), and row upsert with retry logic.
    """

    def __init__(self, db: FormDBBackend, config: FormProcessorConfig) -> None:
        self._db = db
        self._config = config

    def ensure_table(self, template: FormTemplate) -> str:
        """Create the table if it doesn't exist, evolve schema if needed.

        Returns the table name.
        """
        table_name = get_table_name(self._config, template)

        if self._db.table_exists(table_name):
            # Check for schema evolution (new fields in template)
            self.evolve_schema(table_name, template)
        else:
            schema = generate_table_schema(template)
            columns_sql = ", ".join(
                f"[{col}] {col_type}" for col, col_type in schema.items()
            )
            sql = f"CREATE TABLE IF NOT EXISTS [{table_name}] ({columns_sql})"
            self._db.execute_sql(sql)
            logger.info(
                "forms.write.table_created",
                extra={
                    "table_name": table_name,
                    "template_id": template.template_id,
                    "template_name": template.name,
                },
            )

        return table_name

    def evolve_schema(
        self, table_name: str, new_template: FormTemplate
    ) -> list[str]:
        """Add columns for fields in new_template not yet in the table.

        Intentional deviation from spec: queries existing DB columns
        rather than requiring an old_template parameter. The DB is
        the source of truth, making this more robust than comparing
        against a possibly-stale template object.

        Returns list of added column names.
        """
        try:
            existing_columns = set(self._db.get_table_columns(table_name))
            added: list[str] = []

            for field in new_template.fields:
                if field.field_name not in existing_columns:
                    sql_type = FIELD_TYPE_TO_SQL.get(field.field_type, "TEXT")
                    alter_sql = (
                        f"ALTER TABLE [{table_name}] "
                        f"ADD COLUMN [{field.field_name}] {sql_type}"
                    )
                    self._db.execute_sql(alter_sql)
                    added.append(field.field_name)

            if added:
                logger.warning(
                    "forms.write.schema_evolved",
                    extra={
                        "table_name": table_name,
                        "columns_added": added,
                        "column_count": len(added),
                    },
                )

            return added

        except FormIngestException:
            raise
        except Exception as exc:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_DB_SCHEMA_EVOLUTION_FAILED,
                message=f"Schema evolution failed for {table_name}: {exc}",
                stage="output",
                recoverable=False,
            ) from exc

    def write_row(
        self,
        table_name: str,
        extraction: FormExtractionResult,
        ingest_key: str,
        ingest_run_id: str,
    ) -> str:
        """Write a single form row with upsert semantics. Returns _form_id.

        Retries up to ``config.backend_max_retries`` times with
        exponential backoff on failure.
        """
        row = build_row_dict(extraction, self._config, ingest_key, ingest_run_id)
        columns = list(row.keys())
        placeholders = ", ".join("?" for _ in columns)
        columns_sql = ", ".join(f"[{col}]" for col in columns)
        sql = f"INSERT OR REPLACE INTO [{table_name}] ({columns_sql}) VALUES ({placeholders})"
        params = tuple(row[col] for col in columns)

        max_attempts = self._config.backend_max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                self._db.execute_sql(sql, params)
                return extraction.form_id
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    sleep_time = self._config.backend_backoff_base * (2 ** attempt)
                    logger.warning(
                        "forms.write.db_retry",
                        extra={
                            "table_name": table_name,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "retry_delay_s": sleep_time,
                            "error": str(exc),
                        },
                    )
                    time.sleep(sleep_time)

        raise FormIngestException(
            code=FormErrorCode.E_FORM_DB_WRITE_FAILED,
            message=(
                f"DB write failed for {table_name} after {max_attempts} attempts: "
                f"{last_exc}"
            ),
            stage="output",
            recoverable=True,
        )
