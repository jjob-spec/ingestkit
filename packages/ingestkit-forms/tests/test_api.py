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
    TemplateMatch,
    TemplateStatus,
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


# ---------------------------------------------------------------------------
# New API methods (issue #69)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMatchDocument:
    """Tests for FormTemplateAPI.match_document."""

    def test_match_document_delegates_to_matcher(
        self, mock_template_store, form_config
    ):
        """match_document delegates to the injected FormMatcher."""
        from unittest.mock import MagicMock

        mock_matcher = MagicMock()
        mock_matcher.match_document.return_value = [
            TemplateMatch(
                template_id="t1",
                template_name="Form A",
                template_version=1,
                source_format="pdf",
                confidence=0.9,
                per_page_confidence=[0.9],
                matched_features=["layout_grid"],
            )
        ]

        api = FormTemplateAPI(
            store=mock_template_store,
            config=form_config,
            matcher=mock_matcher,
        )
        results = api.match_document("/tmp/form.pdf")
        assert len(results) == 1
        assert results[0].template_id == "t1"
        mock_matcher.match_document.assert_called_once_with("/tmp/form.pdf")

    def test_match_document_raises_without_matcher(
        self, mock_template_store, form_config
    ):
        """Calling match_document without matcher raises FormIngestException."""
        api = FormTemplateAPI(store=mock_template_store, config=form_config)
        with pytest.raises(FormIngestException):
            api.match_document("/tmp/form.pdf")


@pytest.mark.unit
class TestExtractForm:
    """Tests for FormTemplateAPI.extract_form."""

    def test_extract_form_delegates_to_router(
        self, mock_template_store, form_config
    ):
        """extract_form delegates to the injected FormRouter."""
        from unittest.mock import MagicMock

        from ingestkit_forms.models import FormIngestRequest

        mock_router = MagicMock()
        mock_router.extract_form.return_value = MagicMock()

        api = FormTemplateAPI(
            store=mock_template_store,
            config=form_config,
            router=mock_router,
        )
        request = FormIngestRequest(
            file_path="/tmp/form.xlsx", template_id="t1"
        )
        result = api.extract_form(request)
        assert result is not None
        mock_router.extract_form.assert_called_once_with(request)

    def test_extract_form_raises_without_router(
        self, mock_template_store, form_config
    ):
        """Calling extract_form without router raises FormIngestException."""
        from ingestkit_forms.models import FormIngestRequest

        api = FormTemplateAPI(store=mock_template_store, config=form_config)
        request = FormIngestRequest(
            file_path="/tmp/form.xlsx", template_id="t1"
        )
        with pytest.raises(FormIngestException):
            api.extract_form(request)


@pytest.mark.unit
class TestRenderDocument:
    """Tests for FormTemplateAPI.render_document."""

    def test_render_document_returns_png_bytes(
        self, mock_template_store, form_config, tmp_path
    ):
        """render_document returns PNG bytes for an image file."""
        import os
        import random

        from PIL import Image

        # Create a small noisy image -- random pixel data compresses poorly,
        # which keeps the decompression ratio below the safety limit (100x).
        random.seed(42)
        width, height = 100, 100
        pixels = bytes(random.randint(0, 255) for _ in range(width * height * 3))
        img = Image.frombytes("RGB", (width, height), pixels)
        img_path = tmp_path / "render_test.png"
        img.save(str(img_path))

        # Sanity check: compressed size should be a good fraction of decompressed
        compressed = os.path.getsize(str(img_path))
        decompressed = width * height * 3
        assert decompressed / compressed < 100, (
            f"ratio {decompressed / compressed:.1f} still too high"
        )

        api = FormTemplateAPI(store=mock_template_store, config=form_config)
        png_bytes = api.render_document(str(img_path), page=0, dpi=150)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"


@pytest.mark.unit
class TestPreviewExtraction:
    """Tests for FormTemplateAPI.preview_extraction."""

    def test_preview_extraction_raises_without_router(
        self, mock_template_store, form_config
    ):
        """Calling preview_extraction without router raises FormIngestException."""
        api = FormTemplateAPI(store=mock_template_store, config=form_config)
        with pytest.raises(FormIngestException):
            api.preview_extraction("/tmp/form.xlsx", "tmpl-1")


# ---------------------------------------------------------------------------
# Template Lifecycle tests (issue #98)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApproveTemplate:
    """Tests for FormTemplateAPI.approve_template."""

    def test_approve_sets_status_and_fields(self, template_api, sample_image_file):
        """approve_template sets status=APPROVED, approved_by, approved_at."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        assert created.status == TemplateStatus.DRAFT

        approved = template_api.approve_template(created.template_id, "admin-user")
        assert approved.status == TemplateStatus.APPROVED
        assert approved.approved_by == "admin-user"
        assert approved.approved_at is not None

    def test_approve_not_found(self, template_api):
        """approve_template raises E_FORM_TEMPLATE_NOT_FOUND for nonexistent template."""
        with pytest.raises(FormIngestException) as exc_info:
            template_api.approve_template("nonexistent", "admin")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND

    def test_approve_already_approved_raises(self, template_api, sample_image_file):
        """Cannot approve a template that is already APPROVED."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.approve_template(created.template_id, "admin")

        with pytest.raises(FormIngestException) as exc_info:
            template_api.approve_template(created.template_id, "admin2")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_INVALID

    def test_approve_archived_raises(self, template_api, sample_image_file):
        """Cannot approve a template that is ARCHIVED."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.delete_template(created.template_id)

        with pytest.raises(FormIngestException):
            template_api.approve_template(created.template_id, "admin")


@pytest.mark.unit
class TestUpdateTemplateStatus:
    """Tests for status transitions during update_template."""

    def test_field_change_resets_to_draft(self, template_api, sample_image_file):
        """Updating fields resets an APPROVED template back to DRAFT."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.approve_template(created.template_id, "admin")

        new_fields = [
            FieldMapping(
                field_name="new_field",
                field_label="New Field",
                field_type=FieldType.TEXT,
                page_number=0,
                region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
            )
        ]
        update_req = FormTemplateUpdateRequest(fields=new_fields)
        updated = template_api.update_template(created.template_id, update_req)

        assert updated.status == TemplateStatus.DRAFT
        assert updated.approved_by is None
        assert updated.approved_at is None

    def test_metadata_change_preserves_status(self, template_api, sample_image_file):
        """Updating only name/description preserves APPROVED status."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.approve_template(created.template_id, "admin")

        update_req = FormTemplateUpdateRequest(name="Renamed Template")
        updated = template_api.update_template(created.template_id, update_req)

        assert updated.status == TemplateStatus.APPROVED
        assert updated.approved_by == "admin"
        assert updated.approved_at is not None

    def test_page_count_change_resets_to_draft(self, template_api, sample_image_file):
        """Updating page_count resets status to DRAFT."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.approve_template(created.template_id, "admin")

        update_req = FormTemplateUpdateRequest(page_count=3)
        updated = template_api.update_template(created.template_id, update_req)

        assert updated.status == TemplateStatus.DRAFT


@pytest.mark.unit
class TestCreateTemplateStatus:
    """Tests for initial_status in create_template."""

    def test_default_status_is_draft(self, template_api, sample_image_file):
        """Templates are created with DRAFT status by default."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        assert created.status == TemplateStatus.DRAFT

    def test_create_with_approved_status(self, template_api, sample_image_file):
        """Templates can be created with initial_status='approved'."""
        req = FormTemplateCreateRequest(
            name="Pre-approved",
            description="Test",
            source_format=SourceFormat.PDF,
            sample_file_path=sample_image_file,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="f1",
                    field_label="F1",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                )
            ],
            initial_status="approved",
        )
        created = template_api.create_template(req)
        assert created.status == TemplateStatus.APPROVED

    def test_create_with_invalid_status_raises(self, template_api, sample_image_file):
        """Invalid initial_status raises FormIngestException."""
        req = FormTemplateCreateRequest(
            name="Bad Status",
            description="Test",
            source_format=SourceFormat.PDF,
            sample_file_path=sample_image_file,
            page_count=1,
            fields=[
                FieldMapping(
                    field_name="f1",
                    field_label="F1",
                    field_type=FieldType.TEXT,
                    page_number=0,
                    region=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.05),
                )
            ],
            initial_status="invalid_status",
        )
        with pytest.raises(FormIngestException) as exc_info:
            template_api.create_template(req)
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_INVALID


@pytest.mark.unit
class TestDeleteTemplateArchive:
    """Tests for archive behavior in delete_template."""

    def test_delete_sets_archived_status(self, template_api, sample_image_file):
        """delete_template returns template with ARCHIVED status."""
        created = template_api.create_template(
            _make_create_request(sample_file_path=sample_image_file)
        )
        archived = template_api.delete_template(created.template_id)
        assert archived is not None
        assert archived.status == TemplateStatus.ARCHIVED

    def test_delete_not_found_raises(self, template_api):
        """delete_template raises E_FORM_TEMPLATE_NOT_FOUND for nonexistent template."""
        with pytest.raises(FormIngestException) as exc_info:
            template_api.delete_template("nonexistent-id")
        assert exc_info.value.code == FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND


# ---------------------------------------------------------------------------
# Approved-only filtering tests (issue #99)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApprovedOnlyFiltering:
    """Tests for status-based filtering in list_templates and get_all_fingerprints."""

    def test_list_templates_status_filter(self, template_api, sample_image_file):
        """list_templates(status='approved') returns only approved templates."""
        req1 = _make_create_request(
            name="Draft Template", sample_file_path=sample_image_file
        )
        template_api.create_template(req1)

        req2 = _make_create_request(
            name="Approved Template", sample_file_path=sample_image_file
        )
        created2 = template_api.create_template(req2)
        template_api.approve_template(created2.template_id, "admin")

        all_templates = template_api.list_templates()
        assert len(all_templates) == 2

        approved_only = template_api.list_templates(status="approved")
        assert len(approved_only) == 1
        assert approved_only[0].name == "Approved Template"

        draft_only = template_api.list_templates(status="draft")
        assert len(draft_only) == 1
        assert draft_only[0].name == "Draft Template"

    def test_get_all_fingerprints_returns_approved_only(
        self, mock_template_store, form_config, sample_image_file
    ):
        """get_all_fingerprints only returns approved templates."""
        api = FormTemplateAPI(
            store=mock_template_store, config=form_config, renderer=None
        )

        # Create two templates (both have fingerprints from sample_image_file)
        req1 = _make_create_request(
            name="Draft", sample_file_path=sample_image_file
        )
        api.create_template(req1)

        req2 = _make_create_request(
            name="Approved", sample_file_path=sample_image_file
        )
        created2 = api.create_template(req2)
        api.approve_template(created2.template_id, "admin")

        fingerprints = mock_template_store.get_all_fingerprints()
        assert len(fingerprints) == 1
        assert fingerprints[0][1] == "Approved"  # name is second element

    def test_list_templates_no_status_filter(self, template_api, sample_image_file):
        """list_templates() without status returns all active templates."""
        req = _make_create_request(sample_file_path=sample_image_file)
        created = template_api.create_template(req)
        template_api.approve_template(created.template_id, "admin")

        req2 = _make_create_request(
            name="Another Draft", sample_file_path=sample_image_file
        )
        template_api.create_template(req2)

        all_templates = template_api.list_templates()
        assert len(all_templates) == 2
