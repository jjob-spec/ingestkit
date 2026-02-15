"""Plugin API surface for ingestkit-forms.

Provides template CRUD, preview, and matching endpoints consumed
by the orchestration layer (spec section 10).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ingestkit_forms.errors import FormErrorCode, FormIngestException
from ingestkit_forms.security import validate_regex_pattern
from ingestkit_forms.matcher import (
    PageRenderer,
    compute_layout_fingerprint_from_file,
)
from ingestkit_forms.models import (
    ExtractionPreview,
    FormIngestRequest,
    FormProcessingResult,
    FormTemplate,
    FormTemplateCreateRequest,
    FormTemplateUpdateRequest,
    TemplateMatch,
)

if TYPE_CHECKING:
    from ingestkit_forms.config import FormProcessorConfig
    from ingestkit_forms.matcher import FormMatcher
    from ingestkit_forms.protocols import FormTemplateStore
    from ingestkit_forms.router import FormRouter

logger = logging.getLogger("ingestkit_forms")


class FormTemplateAPI:
    """Template CRUD operations for the form ingestor plugin.

    Composes a FormTemplateStore (persistence) and matcher (fingerprinting)
    to implement create, update, delete, get, list, and list_versions.
    """

    def __init__(
        self,
        store: FormTemplateStore,
        config: FormProcessorConfig,
        renderer: PageRenderer | None = None,
        matcher: FormMatcher | None = None,
        router: FormRouter | None = None,
    ) -> None:
        """Initialize the API.

        Args:
            store: FormTemplateStore implementation for persistence.
            config: FormProcessorConfig with fingerprint and matching params.
            renderer: Optional page renderer for non-image file formats.
            matcher: Optional FormMatcher for match_document.
            router: Optional FormRouter for extract_form and preview_extraction.
        """
        self._store = store
        self._config = config
        self._renderer = renderer
        self._matcher = matcher
        self._router = router

    @staticmethod
    def _validate_field_patterns(fields: list) -> None:
        """Validate all regex patterns compile successfully.

        Raises FormIngestException with E_FORM_TEMPLATE_INVALID if any
        validation_pattern is an invalid regex.
        """
        for field in fields:
            if field.validation_pattern is not None:
                error_msg = validate_regex_pattern(field.validation_pattern)
                if error_msg is not None:
                    raise FormIngestException(
                        code=FormErrorCode.E_FORM_TEMPLATE_INVALID,
                        message=(
                            f"Invalid regex pattern for field '{field.field_name}': "
                            f"{error_msg}"
                        ),
                        field_name=field.field_name,
                        stage="api",
                        recoverable=False,
                    )

    def create_template(
        self,
        template_def: FormTemplateCreateRequest,
    ) -> FormTemplate:
        """Create a new template.

        Generates UUID, computes layout fingerprint from sample_file_path,
        sets version=1, persists via store.

        Returns:
            FormTemplate: The created template.

        Raises:
            FormIngestError: E_FORM_TEMPLATE_STORE_UNAVAILABLE if store write fails.
        """
        template_id = str(uuid.uuid4())

        # Validate regex patterns in field mappings
        self._validate_field_patterns(template_def.fields)

        # Compute fingerprint (non-fatal if fails)
        fingerprint: bytes | None = None
        try:
            fingerprint = compute_layout_fingerprint_from_file(
                template_def.sample_file_path,
                self._config,
                self._renderer,
            )
        except Exception as exc:
            logger.warning(
                "Fingerprint computation failed for template '%s': %s. "
                "Template will be created without fingerprint.",
                template_def.name,
                exc,
            )

        now = datetime.now(timezone.utc)
        template = FormTemplate(
            template_id=template_id,
            name=template_def.name,
            description=template_def.description,
            version=1,
            source_format=template_def.source_format,
            page_count=template_def.page_count,
            fields=template_def.fields,
            layout_fingerprint=fingerprint,
            created_at=now,
            updated_at=now,
            created_by=template_def.created_by,
            tenant_id=template_def.tenant_id,
        )

        self._store.save_template(template)
        logger.info(
            "Created template '%s' (id=%s, v1)", template.name, template_id
        )
        return template

    def update_template(
        self,
        template_id: str,
        template_def: FormTemplateUpdateRequest,
    ) -> FormTemplate:
        """Update a template, creating a new version.

        Loads the latest version, increments version, applies partial updates,
        recomputes fingerprint if new sample_file_path is provided.

        Returns:
            FormTemplate: The new version.

        Raises:
            FormIngestError: E_FORM_TEMPLATE_NOT_FOUND if template doesn't exist.
        """
        existing = self._store.get_template(template_id)
        if existing is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=f"Template '{template_id}' not found",
                stage="api",
                recoverable=False,
            )

        new_version = existing.version + 1

        # Apply partial updates (only non-None fields)
        name = template_def.name if template_def.name is not None else existing.name
        description = (
            template_def.description
            if template_def.description is not None
            else existing.description
        )
        page_count = (
            template_def.page_count
            if template_def.page_count is not None
            else existing.page_count
        )
        fields = (
            template_def.fields
            if template_def.fields is not None
            else existing.fields
        )

        # Validate regex patterns if new fields provided
        if template_def.fields is not None:
            self._validate_field_patterns(fields)

        # Recompute fingerprint if new sample file provided
        fingerprint = existing.layout_fingerprint
        if template_def.sample_file_path is not None:
            try:
                fingerprint = compute_layout_fingerprint_from_file(
                    template_def.sample_file_path,
                    self._config,
                    self._renderer,
                )
            except Exception as exc:
                logger.warning(
                    "Fingerprint recomputation failed for template '%s' v%d: %s. "
                    "Keeping existing fingerprint.",
                    template_id,
                    new_version,
                    exc,
                )

        now = datetime.now(timezone.utc)
        updated = FormTemplate(
            template_id=template_id,
            name=name,
            description=description,
            version=new_version,
            source_format=existing.source_format,  # immutable
            page_count=page_count,
            fields=fields,
            layout_fingerprint=fingerprint,
            thumbnail=existing.thumbnail,
            created_at=existing.created_at,  # preserve original creation time
            updated_at=now,
            created_by=existing.created_by,
            tenant_id=existing.tenant_id,
        )

        self._store.save_template(updated)
        logger.info(
            "Updated template '%s' (id=%s, v%d)",
            updated.name,
            template_id,
            new_version,
        )
        return updated

    def delete_template(
        self,
        template_id: str,
        version: int | None = None,
    ) -> None:
        """Soft-delete a template or specific version.

        Delegates to store.delete_template(). Raises if template not found.
        """
        self._store.delete_template(template_id, version)
        logger.info(
            "Deleted template %s (version=%s)", template_id, version or "all"
        )

    def get_template(
        self,
        template_id: str,
        version: int | None = None,
    ) -> FormTemplate:
        """Get a specific template by ID.

        If version is None, returns the latest version.
        Raises E_FORM_TEMPLATE_NOT_FOUND if not found.
        """
        template = self._store.get_template(template_id, version)
        if template is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=(
                    f"Template '{template_id}' "
                    f"version {version or 'latest'} not found"
                ),
                stage="api",
                recoverable=False,
            )
        return template

    def list_templates(
        self,
        tenant_id: str | None = None,
        source_format: str | None = None,
    ) -> list[FormTemplate]:
        """List all active templates, optionally filtered by tenant and format.

        Returns the latest version of each template.
        """
        return self._store.list_templates(
            tenant_id=tenant_id,
            source_format=source_format,
            active_only=True,
        )

    def list_template_versions(
        self,
        template_id: str,
    ) -> list[FormTemplate]:
        """List all versions of a template, ordered by version descending."""
        versions = self._store.list_versions(template_id)
        if len(versions) == 0:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_TEMPLATE_NOT_FOUND,
                message=f"Template '{template_id}' not found",
                stage="api",
                recoverable=False,
            )
        return versions

    # ------------------------------------------------------------------
    # Plugin API operations 7-10 (spec section 9.1)
    # ------------------------------------------------------------------

    def render_document(
        self,
        file_path: str,
        page: int = 0,
        dpi: int = 150,
    ) -> bytes:
        """Render a document page as PNG bytes.

        Uses the extractors rendering module for images and PDFs.

        Args:
            file_path: Path to the document file.
            page: 0-indexed page number.
            dpi: Target DPI for rendering.

        Returns:
            PNG-encoded bytes of the rendered page.

        Raises:
            FormIngestException if rendering fails.
        """
        import io

        from ingestkit_forms.extractors._rendering import get_page_image

        try:
            image = get_page_image(file_path, page, dpi)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
        except FormIngestException:
            raise
        except Exception as exc:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message=f"Document rendering failed: {exc}",
                stage="api",
                recoverable=False,
            ) from exc

    def preview_extraction(
        self,
        file_path: str,
        template_id: str,
        template_version: int | None = None,
    ) -> ExtractionPreview:
        """Preview extraction results without persisting.

        Runs the extraction pipeline through the router's internal
        components (extractor + confidence) but skips dual-write.

        Args:
            file_path: Path to the document.
            template_id: Template to apply.
            template_version: Specific version (None = latest).

        Returns:
            ExtractionPreview with extracted fields and confidence.

        Raises:
            FormIngestException if router is not configured or extraction fails.
        """
        if self._router is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message="FormRouter not configured -- cannot preview extraction",
                stage="api",
                recoverable=False,
            )

        template = self.get_template(template_id, template_version)
        method, fields, _duration = self._router._run_extraction(
            file_path, template
        )
        fields, overall_confidence, warnings = (
            self._router._apply_confidence_and_vlm(fields, template, file_path)
        )

        return ExtractionPreview(
            template_id=template.template_id,
            template_name=template.name,
            template_version=template.version,
            fields=fields,
            overall_confidence=overall_confidence,
            extraction_method=method,
            warnings=warnings,
        )

    def match_document(
        self,
        file_path: str,
        tenant_id: str | None = None,
    ) -> list[TemplateMatch]:
        """Match a document against registered templates.

        Args:
            file_path: Path to the document.
            tenant_id: Optional tenant filter (unused by matcher directly,
                reserved for future filtering).

        Returns:
            List of TemplateMatch sorted by confidence descending.

        Raises:
            FormIngestException if matcher is not configured.
        """
        if self._matcher is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message="FormMatcher not configured -- cannot match document",
                stage="api",
                recoverable=False,
            )

        return self._matcher.match_document(file_path)

    def extract_form(
        self,
        request: FormIngestRequest,
    ) -> FormProcessingResult | None:
        """Execute the full form extraction pipeline.

        Delegates to the router's ``extract_form`` method.

        Args:
            request: FormIngestRequest with file path and optional template_id.

        Returns:
            FormProcessingResult on success, None on graceful fallthrough.

        Raises:
            FormIngestException if router is not configured.
        """
        if self._router is None:
            raise FormIngestException(
                code=FormErrorCode.E_FORM_EXTRACTION_FAILED,
                message="FormRouter not configured -- cannot extract form",
                stage="api",
                recoverable=False,
            )

        return self._router.extract_form(request)
