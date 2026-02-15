"""Tests for the plugin API surface."""

from __future__ import annotations

import pytest

from ingestkit_forms.api import FormTemplateAPI
from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.models import (
    BoundingBox,
    FieldMapping,
    FieldType,
    FormTemplateCreateRequest,
    FormTemplateUpdateRequest,
    SourceFormat,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_create_request(
    name: str = "W-4 2026",
    sample_file_path: str = "/tmp/sample.png",
    source_format: SourceFormat = SourceFormat.PDF,
    tenant_id: str | None = "tenant-a",
) -> FormTemplateCreateRequest:
    """Factory for FormTemplateCreateRequest."""
    return FormTemplateCreateRequest(
        name=name,
        description="Test template",
        source_format=source_format,
        sample_file_path=sample_file_path,
        page_count=1,
        fields=[
            FieldMapping(
                field_name="employee_name",
                field_label="Employee Name",
                field_type=FieldType.TEXT,
                page_number=0,
                region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
            )
        ],
        tenant_id=tenant_id,
        created_by="test",
    )


@pytest.fixture()
def template_api(mock_template_store, form_config):
    """Create a FormTemplateAPI with mock store and test config."""
    return FormTemplateAPI(
        store=mock_template_store,
        config=form_config,
        renderer=None,
    )


# ---------------------------------------------------------------------------
# Create tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateTemplate:
    """Tests for FormTemplateAPI.create_template."""

    def test_create_template(self, template_api, sample_image_file):
        """Creates template with valid request, version=1, UUID generated, fingerprint computed."""
        req = _make_create_request(sample_file_path=sample_image_file)
        template = template_api.create_template(req)

        assert template.version == 1
        assert template.name == "W-4 2026"
        assert template.tenant_id == "tenant-a"
        assert len(template.template_id) == 36  # UUID format
        assert template.layout_fingerprint is not None
        assert len(template.layout_fingerprint) > 0

    def test_create_template_fingerprint_failure_still_succeeds(
        self, template_api
    ):
        """If fingerprint computation raises, template is created with layout_fingerprint=None."""
        req = _make_create_request(sample_file_path="/nonexistent/file.png")
        template = template_api.create_template(req)

        assert template.version == 1
        assert template.layout_fingerprint is None
        assert template.name == "W-4 2026"


# ---------------------------------------------------------------------------
# Update tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUpdateTemplate:
    """Tests for FormTemplateAPI.update_template."""

    def _create_initial(self, template_api, sample_image_file):
        """Helper to create an initial template."""
        req = _make_create_request(sample_file_path=sample_image_file)
        return template_api.create_template(req)

    def test_update_template(self, template_api, sample_image_file):
        """Update with new name, version increments, previous version retained."""
        initial = self._create_initial(template_api, sample_image_file)
        update_req = FormTemplateUpdateRequest(name="W-4 2027")
        updated = template_api.update_template(initial.template_id, update_req)

        assert updated.version == 2
        assert updated.name == "W-4 2027"
        # Previous version still accessible
        v1 = template_api.get_template(initial.template_id, version=1)
        assert v1.name == "W-4 2026"

    def test_update_with_new_sample(self, template_api, sample_image_file, tmp_path):
        """Update with sample_file_path recomputes fingerprint."""
        initial = self._create_initial(template_api, sample_image_file)

        # Create a structurally different sample image (checkerboard pattern)
        from PIL import Image, ImageDraw

        new_img = Image.new("L", (1200, 1600), color=255)
        draw = ImageDraw.Draw(new_img)
        # Fill entire top half with black -- very different from original form
        draw.rectangle([0, 0, 1200, 800], fill=0)
        new_path = tmp_path / "new_sample.png"
        new_img.save(str(new_path))

        update_req = FormTemplateUpdateRequest(sample_file_path=str(new_path))
        updated = template_api.update_template(initial.template_id, update_req)

        assert updated.version == 2
        assert updated.layout_fingerprint is not None

    def test_update_not_found(self, template_api):
        """Update on nonexistent template raises FormIngestError."""
        update_req = FormTemplateUpdateRequest(name="New Name")
        with pytest.raises(FormIngestException) as exc_info:
            template_api.update_template("nonexistent-id", update_req)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND

    def test_update_preserves_immutables(self, template_api, sample_image_file):
        """source_format, tenant_id, created_at, created_by are NOT changed by update."""
        initial = self._create_initial(template_api, sample_image_file)
        update_req = FormTemplateUpdateRequest(name="Updated Name")
        updated = template_api.update_template(initial.template_id, update_req)

        assert updated.source_format == initial.source_format
        assert updated.tenant_id == initial.tenant_id
        assert updated.created_at == initial.created_at
        assert updated.created_by == initial.created_by


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteTemplate:
    """Tests for FormTemplateAPI.delete_template."""

    def test_delete_all(self, template_api, sample_image_file):
        """delete_template(id) makes subsequent get_template raise."""
        initial = template_api.create_template(
            _make_create_request(sample_file_path=sample_image_file)
        )
        template_api.delete_template(initial.template_id)

        with pytest.raises(FormIngestException) as exc_info:
            template_api.get_template(initial.template_id)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND

    def test_delete_specific_version(self, template_api, sample_image_file):
        """delete_template(id, version=1) deletes only that version."""
        initial = template_api.create_template(
            _make_create_request(sample_file_path=sample_image_file)
        )
        update_req = FormTemplateUpdateRequest(name="v2")
        template_api.update_template(initial.template_id, update_req)

        template_api.delete_template(initial.template_id, version=1)

        # v1 is gone
        with pytest.raises(FormIngestException):
            template_api.get_template(initial.template_id, version=1)

        # v2 still accessible
        v2 = template_api.get_template(initial.template_id, version=2)
        assert v2.name == "v2"


# ---------------------------------------------------------------------------
# Get tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetTemplate:
    """Tests for FormTemplateAPI.get_template."""

    def test_get_found(self, template_api, sample_image_file):
        """Returns the template from store."""
        initial = template_api.create_template(
            _make_create_request(sample_file_path=sample_image_file)
        )
        result = template_api.get_template(initial.template_id)
        assert result.template_id == initial.template_id

    def test_get_not_found(self, template_api):
        """Raises FormIngestError with E_FORM_TEMPLATE_NOT_FOUND."""
        with pytest.raises(FormIngestException) as exc_info:
            template_api.get_template("nonexistent")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND


# ---------------------------------------------------------------------------
# List tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListTemplates:
    """Tests for FormTemplateAPI.list_templates."""

    def test_list_templates(self, template_api, sample_image_file):
        """Delegates to store with correct filters."""
        template_api.create_template(
            _make_create_request(
                name="Template A",
                sample_file_path=sample_image_file,
                tenant_id="tenant-a",
            )
        )
        template_api.create_template(
            _make_create_request(
                name="Template B",
                sample_file_path=sample_image_file,
                tenant_id="tenant-b",
            )
        )

        all_templates = template_api.list_templates()
        assert len(all_templates) == 2

        filtered = template_api.list_templates(tenant_id="tenant-a")
        assert len(filtered) == 1
        assert filtered[0].name == "Template A"


@pytest.mark.unit
class TestListTemplateVersions:
    """Tests for FormTemplateAPI.list_template_versions."""

    def test_list_versions(self, template_api, sample_image_file):
        """Returns versions from store ordered descending."""
        initial = template_api.create_template(
            _make_create_request(sample_file_path=sample_image_file)
        )
        template_api.update_template(
            initial.template_id,
            FormTemplateUpdateRequest(name="v2"),
        )

        versions = template_api.list_template_versions(initial.template_id)
        assert len(versions) == 2
        assert versions[0].version == 2
        assert versions[1].version == 1

    def test_list_versions_not_found(self, template_api):
        """Raises FormIngestError if no versions exist."""
        with pytest.raises(FormIngestException) as exc_info:
            template_api.list_template_versions("nonexistent")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND
