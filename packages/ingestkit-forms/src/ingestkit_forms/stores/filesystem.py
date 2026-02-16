"""Filesystem-based FormTemplateStore implementation.

Persists form templates as JSON files with directory-based organization:
    {base_path}/{template_id}/v{version}.json  -- template data
    {base_path}/{template_id}/_meta.json        -- deletion/version metadata

Implements the FormTemplateStore protocol via structural subtyping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import FormTemplate

logger = logging.getLogger("ingestkit_forms")


class _TemplateMeta(BaseModel):
    """Internal metadata for a template tracked by the filesystem store.

    Stored as _meta.json per template_id directory.
    NOT part of the public API or the FormTemplate model.
    """

    template_id: str
    latest_version: int = 1
    deleted_versions: list[int] = Field(default_factory=list)
    all_deleted: bool = False


class FileSystemTemplateStore:
    """Filesystem-based FormTemplateStore implementation.

    Directory structure:
        {base_path}/{template_id}/v{version}.json  -- template data
        {base_path}/{template_id}/_meta.json        -- deletion/version metadata

    Implements FormTemplateStore protocol (structural subtyping).
    Passes isinstance(store, FormTemplateStore) check.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize the store.

        Args:
            base_path: Root directory for template storage.
                Created if it does not exist.
        """
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def save_template(self, template: FormTemplate) -> None:
        """Persist a template (insert or update).

        Creates directory {base_path}/{template_id}/ if needed.
        Writes template as v{version}.json.
        Updates _meta.json with latest_version.
        """
        template_dir = self._base_path / template.template_id
        template_dir.mkdir(parents=True, exist_ok=True)

        # Serialize template
        data = template.model_dump(mode="json")

        # Write version file
        version_file = template_dir / f"v{template.version}.json"
        version_file.write_text(json.dumps(data, indent=2))

        # Update metadata
        meta = self._load_meta(template.template_id)
        if meta is None:
            meta = _TemplateMeta(
                template_id=template.template_id,
                latest_version=template.version,
            )
        else:
            meta.latest_version = max(meta.latest_version, template.version)
        self._save_meta(template.template_id, meta)

    def get_template(
        self, template_id: str, version: int | None = None
    ) -> FormTemplate | None:
        """Retrieve a template by ID. None if not found.

        If version is None, returns the latest version.
        Returns None if the template/version is soft-deleted.
        """
        meta = self._load_meta(template_id)
        if meta is None:
            return None

        if version is None:
            # Find the latest non-deleted version
            if meta.all_deleted:
                return None
            version = meta.latest_version
            # Walk backwards to find latest non-deleted version
            while version > 0 and version in meta.deleted_versions:
                version -= 1
            if version == 0:
                return None

        if meta.all_deleted or version in meta.deleted_versions:
            return None

        file_path = self._base_path / template_id / f"v{version}.json"
        if not file_path.exists():
            return None

        return self._load_template_file(file_path)

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
        active_only: bool = True,
        status: str | None = None,
    ) -> list[FormTemplate]:
        """List templates matching the filters.

        Returns the latest version of each template.
        If active_only=True (default), excludes soft-deleted templates.
        If status is provided, only templates with that status are returned.
        """
        results: list[FormTemplate] = []

        if not self._base_path.exists():
            return results

        for template_dir in sorted(self._base_path.iterdir()):
            if not template_dir.is_dir():
                continue

            template_id = template_dir.name
            meta = self._load_meta(template_id)
            if meta is None:
                continue

            if active_only and meta.all_deleted:
                continue

            template = self.get_template(template_id)
            if template is None:
                continue

            if tenant_id is not None and template.tenant_id != tenant_id:
                continue

            if source_format is not None and template.source_format.value != source_format:
                continue

            if status is not None and template.status.value != status:
                continue

            results.append(template)

        return results

    def list_versions(self, template_id: str) -> list[FormTemplate]:
        """List all versions of a template, ordered by version descending.

        Includes soft-deleted versions. The spec (line 1262) says 'list all versions'
        which is intentional for admin/audit views. Callers can check deletion state
        via the store's metadata if needed.
        """
        template_dir = self._base_path / template_id
        if not template_dir.exists():
            return []

        templates: list[FormTemplate] = []
        for f in sorted(template_dir.iterdir()):
            if f.name.startswith("v") and f.name.endswith(".json"):
                templates.append(self._load_template_file(f))

        # Sort by version descending
        templates.sort(key=lambda t: t.version, reverse=True)
        return templates

    def delete_template(
        self, template_id: str, version: int | None = None
    ) -> None:
        """Soft-delete a template or specific version.

        If version is None, marks all versions as deleted (sets all_deleted=True).
        If version is specified, adds that version to deleted_versions.
        Files are NOT removed from disk (audit trail preservation).
        """
        meta = self._load_meta(template_id)
        if meta is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=f"Template '{template_id}' not found",
                stage="template_store",
                recoverable=False,
            )

        if version is None:
            meta.all_deleted = True
        else:
            # Verify version file exists
            version_file = self._base_path / template_id / f"v{version}.json"
            if not version_file.exists():
                raise FormIngestException(
                    code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                    message=f"Template '{template_id}' version {version} not found",
                    stage="template_store",
                    recoverable=False,
                )
            if version not in meta.deleted_versions:
                meta.deleted_versions.append(version)
            # Check if all versions are now deleted
            all_versions = set(range(1, meta.latest_version + 1))
            if all_versions == set(meta.deleted_versions):
                meta.all_deleted = True

        self._save_meta(template_id, meta)

    def get_all_fingerprints(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[tuple[str, str, int, bytes]]:
        """Return (template_id, name, version, fingerprint) for approved templates.

        Used by the matcher for efficient batch comparison.
        Only returns approved templates with non-None layout_fingerprint.
        """
        templates = self.list_templates(
            tenant_id=tenant_id,
            source_format=source_format,
            active_only=True,
            status="approved",
        )
        results: list[tuple[str, str, int, bytes]] = []
        for t in templates:
            if t.layout_fingerprint is not None:
                results.append(
                    (t.template_id, t.name, t.version, t.layout_fingerprint)
                )
        return results

    # --- Private helpers ---

    def _load_meta(self, template_id: str) -> _TemplateMeta | None:
        """Load _meta.json for a template. Returns None if not found."""
        meta_path = self._base_path / template_id / "_meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return _TemplateMeta(**data)

    def _save_meta(self, template_id: str, meta: _TemplateMeta) -> None:
        """Write _meta.json for a template."""
        meta_path = self._base_path / template_id / "_meta.json"
        meta_path.write_text(json.dumps(meta.model_dump(), indent=2))

    def _load_template_file(self, file_path: Path) -> FormTemplate:
        """Load a single template JSON file and return a FormTemplate."""
        data = json.loads(file_path.read_text())
        # Deserialize hex-encoded bytes fields
        if data.get("layout_fingerprint"):
            data["layout_fingerprint"] = bytes.fromhex(data["layout_fingerprint"])
        if data.get("thumbnail"):
            data["thumbnail"] = bytes.fromhex(data["thumbnail"])
        return FormTemplate(**data)
