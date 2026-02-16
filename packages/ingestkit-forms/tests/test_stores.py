"""Tests for template store implementations."""

from __future__ import annotations

import pytest

from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    BoundingBox,
    FieldMapping,
    FieldType,
    FormTemplate,
    SourceFormat,
    TemplateStatus,
)
from ingestkit_forms.protocols import FormTemplateStore
from ingestkit_forms.stores.filesystem import FileSystemTemplateStore


def _make_template(
    template_id: str = "tmpl-001",
    name: str = "Test Template",
    version: int = 1,
    source_format: SourceFormat = SourceFormat.PDF,
    page_count: int = 1,
    tenant_id: str | None = "tenant-a",
    layout_fingerprint: bytes | None = None,
    status: TemplateStatus = TemplateStatus.DRAFT,
) -> FormTemplate:
    """Factory for FormTemplate test instances."""
    return FormTemplate(
        template_id=template_id,
        name=name,
        version=version,
        source_format=source_format,
        page_count=page_count,
        fields=[
            FieldMapping(
                field_name="field_1",
                field_label="Field 1",
                field_type=FieldType.TEXT,
                page_number=0,
                region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
            )
        ],
        tenant_id=tenant_id,
        layout_fingerprint=layout_fingerprint,
        status=status,
    )


@pytest.mark.unit
class TestFileSystemTemplateStore:
    """Tests for FileSystemTemplateStore."""

    def _make_store(self, tmp_path) -> FileSystemTemplateStore:
        return FileSystemTemplateStore(str(tmp_path / "templates"))

    def test_protocol_compliance(self, tmp_path):
        """isinstance(store, FormTemplateStore) is True."""
        store = self._make_store(tmp_path)
        assert isinstance(store, FormTemplateStore)

    def test_save_and_get_template(self, tmp_path):
        """Save a template, get it back, all fields match."""
        store = self._make_store(tmp_path)
        template = _make_template()
        store.save_template(template)

        retrieved = store.get_template("tmpl-001")
        assert retrieved is not None
        assert retrieved.template_id == "tmpl-001"
        assert retrieved.name == "Test Template"
        assert retrieved.version == 1
        assert retrieved.source_format == SourceFormat.PDF
        assert retrieved.tenant_id == "tenant-a"

    def test_get_template_latest_version(self, tmp_path):
        """Save v1 and v2, get_template(id) returns v2."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(version=1))
        store.save_template(_make_template(version=2, name="Updated"))

        retrieved = store.get_template("tmpl-001")
        assert retrieved is not None
        assert retrieved.version == 2
        assert retrieved.name == "Updated"

    def test_get_template_specific_version(self, tmp_path):
        """Save v1 and v2, get_template(id, version=1) returns v1."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(version=1, name="Original"))
        store.save_template(_make_template(version=2, name="Updated"))

        retrieved = store.get_template("tmpl-001", version=1)
        assert retrieved is not None
        assert retrieved.version == 1
        assert retrieved.name == "Original"

    def test_get_template_not_found(self, tmp_path):
        """get_template('nonexistent') returns None."""
        store = self._make_store(tmp_path)
        assert store.get_template("nonexistent") is None

    def test_list_templates_returns_latest(self, tmp_path):
        """Save two templates with multiple versions, list returns one entry per template (latest)."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(template_id="tmpl-001", version=1))
        store.save_template(_make_template(template_id="tmpl-001", version=2))
        store.save_template(_make_template(template_id="tmpl-002", version=1, name="Other"))

        templates = store.list_templates()
        assert len(templates) == 2
        versions = {t.template_id: t.version for t in templates}
        assert versions["tmpl-001"] == 2
        assert versions["tmpl-002"] == 1

    def test_list_templates_tenant_filter(self, tmp_path):
        """Two templates with different tenant_id, filter returns correct subset."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(template_id="tmpl-001", tenant_id="tenant-a"))
        store.save_template(_make_template(template_id="tmpl-002", tenant_id="tenant-b"))

        templates = store.list_templates(tenant_id="tenant-a")
        assert len(templates) == 1
        assert templates[0].template_id == "tmpl-001"

    def test_list_templates_format_filter(self, tmp_path):
        """Two templates with different source_format, filter returns correct subset."""
        store = self._make_store(tmp_path)
        store.save_template(
            _make_template(template_id="tmpl-001", source_format=SourceFormat.PDF)
        )
        store.save_template(
            _make_template(template_id="tmpl-002", source_format=SourceFormat.IMAGE)
        )

        templates = store.list_templates(source_format="pdf")
        assert len(templates) == 1
        assert templates[0].template_id == "tmpl-001"

    def test_list_templates_active_only(self, tmp_path):
        """Delete one template, list_templates(active_only=True) excludes it."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(template_id="tmpl-001"))
        store.save_template(_make_template(template_id="tmpl-002", name="Other"))
        store.delete_template("tmpl-001")

        active = store.list_templates(active_only=True)
        assert len(active) == 1
        assert active[0].template_id == "tmpl-002"

    def test_list_versions(self, tmp_path):
        """Save three versions, list_versions returns all three ordered descending."""
        store = self._make_store(tmp_path)
        for v in [1, 2, 3]:
            store.save_template(_make_template(version=v, name=f"v{v}"))

        versions = store.list_versions("tmpl-001")
        assert len(versions) == 3
        assert [v.version for v in versions] == [3, 2, 1]

    def test_list_versions_not_found(self, tmp_path):
        """list_versions('nonexistent') returns empty list."""
        store = self._make_store(tmp_path)
        assert store.list_versions("nonexistent") == []

    def test_delete_all_versions(self, tmp_path):
        """delete_template(id) with no version soft-deletes all."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(version=1))
        store.save_template(_make_template(version=2))
        store.delete_template("tmpl-001")

        assert store.get_template("tmpl-001") is None
        assert store.get_template("tmpl-001", version=1) is None
        assert store.get_template("tmpl-001", version=2) is None

    def test_delete_specific_version(self, tmp_path):
        """delete_template(id, version=1) marks v1 deleted. get_template returns latest non-deleted."""
        store = self._make_store(tmp_path)
        store.save_template(_make_template(version=1, name="v1"))
        store.save_template(_make_template(version=2, name="v2"))
        store.delete_template("tmpl-001", version=2)

        assert store.get_template("tmpl-001", version=2) is None
        latest = store.get_template("tmpl-001")
        assert latest is not None
        assert latest.version == 1

    def test_delete_nonexistent_raises(self, tmp_path):
        """delete_template('nonexistent') raises FormIngestError."""
        store = self._make_store(tmp_path)
        with pytest.raises(FormIngestException) as exc_info:
            store.delete_template("nonexistent")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND

    def test_get_all_fingerprints(self, tmp_path):
        """Save templates with/without fingerprints. Only those with fingerprints appear."""
        store = self._make_store(tmp_path)
        store.save_template(
            _make_template(
                template_id="tmpl-001",
                layout_fingerprint=b"\x00\x01\x02\x03",
                status=TemplateStatus.APPROVED,
            )
        )
        store.save_template(
            _make_template(
                template_id="tmpl-002",
                layout_fingerprint=None,
                status=TemplateStatus.APPROVED,
            )
        )

        fps = store.get_all_fingerprints()
        assert len(fps) == 1
        tid, name, version, fp = fps[0]
        assert tid == "tmpl-001"
        assert fp == b"\x00\x01\x02\x03"

    def test_get_all_fingerprints_filters(self, tmp_path):
        """Verify tenant_id and source_format filters work."""
        store = self._make_store(tmp_path)
        store.save_template(
            _make_template(
                template_id="tmpl-001",
                tenant_id="tenant-a",
                layout_fingerprint=b"\x00\x01",
                status=TemplateStatus.APPROVED,
            )
        )
        store.save_template(
            _make_template(
                template_id="tmpl-002",
                tenant_id="tenant-b",
                layout_fingerprint=b"\x02\x03",
                status=TemplateStatus.APPROVED,
            )
        )

        fps = store.get_all_fingerprints(tenant_id="tenant-a")
        assert len(fps) == 1
        assert fps[0][0] == "tmpl-001"

    def test_bytes_round_trip(self, tmp_path):
        """Save template with layout_fingerprint bytes, retrieve, bytes match exactly."""
        store = self._make_store(tmp_path)
        original_fp = b"\x00\x01\x02\x03\xff\xfe\xfd"
        store.save_template(
            _make_template(layout_fingerprint=original_fp)
        )

        retrieved = store.get_template("tmpl-001")
        assert retrieved is not None
        assert retrieved.layout_fingerprint == original_fp

    def test_persistence_across_instances(self, tmp_path):
        """Save with one store instance, create new store with same path, retrieve succeeds."""
        path = str(tmp_path / "templates")
        store1 = FileSystemTemplateStore(path)
        store1.save_template(_make_template())

        store2 = FileSystemTemplateStore(path)
        retrieved = store2.get_template("tmpl-001")
        assert retrieved is not None
        assert retrieved.name == "Test Template"
