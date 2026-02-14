"""Tier 2/3 LLM-based PDF classifier with schema validation.

Generates a structural summary from a DocumentProfile, sends it to an LLM
backend via a classification prompt, validates the response against a
Pydantic schema, and returns a ClassificationResult.

The classifier does NOT handle tier escalation logic -- that is the
responsibility of the router.  It classifies at the specific tier
requested.  ConnectionError is propagated to the caller so the router
can degrade to Tier 1.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ingestkit_pdf.config import PDFProcessorConfig
from ingestkit_pdf.errors import ErrorCode, IngestError
from ingestkit_pdf.models import (
    ClassificationResult,
    ClassificationTier,
    DocumentProfile,
    PageProfile,
    PageType,
    PDFType,
)
from ingestkit_pdf.protocols import LLMBackend

logger = logging.getLogger("ingestkit_pdf")


# ---------------------------------------------------------------------------
# LLM Response Schema
# ---------------------------------------------------------------------------


class PageTypeEntry(BaseModel):
    """Strict sub-model for validating individual page type entries."""

    page: int
    type: Literal[
        "text", "scanned", "table_heavy", "form",
        "mixed", "blank", "vector_only", "toc",
    ]


class LLMClassificationResponse(BaseModel):
    """Schema for validating LLM classification output.

    The ``type`` field uses Literal string values matching the ``PDFType``
    enum values.  Confidence bounds are checked manually (not via Field
    constraints) to allow clamping instead of rejection.
    """

    type: Literal["text_native", "scanned", "complex"]
    confidence: float  # NO ge/le constraints -- checked manually to allow clamping
    reasoning: str = Field(min_length=1)
    page_types: list[PageTypeEntry] | None = None


# ---------------------------------------------------------------------------
# LLM Classifier
# ---------------------------------------------------------------------------


class PDFLLMClassifier:
    """Tier 2/3 LLM-based PDF classifier with schema validation.

    Generates a structural summary from a DocumentProfile, sends it to an
    LLM backend via a classification prompt, validates the response against
    a Pydantic schema, and returns a ClassificationResult.

    The classifier does NOT handle tier escalation logic -- that is the
    responsibility of the router.  It classifies at the specific tier
    requested.
    """

    def __init__(self, llm: LLMBackend, config: PDFProcessorConfig) -> None:
        self._llm = llm
        self._config = config

    # -- public API ----------------------------------------------------------

    def classify(
        self,
        profile: DocumentProfile,
        tier: ClassificationTier,
    ) -> ClassificationResult:
        """Classify a PDF file at the specified LLM tier.

        Args:
            profile: The structural profile of the PDF file.
            tier: Which tier to run (``LLM_BASIC`` or ``LLM_REASONING``).

        Returns:
            A :class:`ClassificationResult` with PDF type, confidence,
            tier information, and reasoning.

        Raises:
            ValueError: If ``tier`` is ``ClassificationTier.RULE_BASED``.
            ConnectionError: If the LLM backend is unreachable (propagated
                to the caller for outage degradation by the router).
        """
        # 1. Validate tier parameter
        if tier == ClassificationTier.RULE_BASED:
            raise ValueError("PDFLLMClassifier does not handle rule_based tier")

        # 2. Select model based on tier
        if tier == ClassificationTier.LLM_BASIC:
            model = self._config.classification_model
        else:  # LLM_REASONING
            model = self._config.reasoning_model

        # 3. Generate structural summary (PII-safe by default)
        summary = self._generate_structural_summary(profile)

        # 4. Build classification prompt
        prompt = self._build_classification_prompt(summary, tier)

        # 5. Attempt classification with retry loop
        errors: list[IngestError] = []
        max_attempts = 2  # 1 original + 1 retry

        for attempt in range(max_attempts):
            if attempt > 0:
                errors.append(
                    IngestError(
                        code=ErrorCode.W_LLM_RETRY,
                        message=f"Retrying LLM classification (attempt {attempt + 1}/{max_attempts})",
                        stage="classify",
                        recoverable=True,
                    )
                )

            # 5a. Call LLM backend
            try:
                raw_dict = self._llm.classify(
                    prompt=prompt,
                    model=model,
                    temperature=self._config.llm_temperature,
                    timeout=self._config.backend_timeout_seconds,
                )
            except json.JSONDecodeError as exc:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_LLM_MALFORMED_JSON,
                        message=f"LLM returned unparseable JSON: {exc}",
                        stage="classify",
                        recoverable=True,
                    )
                )
                prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Respond with ONLY a JSON object."
                )
                continue
            except TimeoutError:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_LLM_TIMEOUT,
                        message=f"LLM backend timed out after {self._config.backend_timeout_seconds}s",
                        stage="classify",
                        recoverable=True,
                    )
                )
                continue
            except ConnectionError:
                # Propagate to caller -- router handles outage degradation
                raise
            except Exception as exc:
                errors.append(
                    IngestError(
                        code=ErrorCode.E_LLM_MALFORMED_JSON,
                        message=f"LLM backend error: {exc}",
                        stage="classify",
                        recoverable=True,
                    )
                )
                prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Respond with ONLY a JSON object."
                )
                continue

            # 5b. Optionally log LLM prompt/response at DEBUG
            if self._config.log_llm_prompts:
                logger.debug("LLM prompt:\n%s", self._redact(prompt))
                logger.debug("LLM response: %s", self._redact(str(raw_dict)))

            # 5c. Validate and parse response
            parsed, validation_errors = self._validate_and_parse_response(
                raw_dict, profile
            )
            errors.extend(validation_errors)

            if parsed is None:
                # Validation failed -- append correction hint and retry
                prompt = (
                    prompt
                    + '\n\nIMPORTANT: Your previous response had schema errors. '
                    'Ensure \'type\' is one of "text_native", "scanned", '
                    '"complex". Ensure \'confidence\' is a float. Ensure \'reasoning\' '
                    'is a non-empty string. Respond with ONLY a JSON object.'
                )
                continue

            # 5d. Convert to ClassificationResult and return
            return self._to_classification_result(parsed, tier)

        # 6. All attempts exhausted -- fail closed
        errors.append(
            IngestError(
                code=ErrorCode.E_CLASSIFY_INCONCLUSIVE,
                message="LLM classification failed after all retry attempts",
                stage="classify",
                recoverable=False,
            )
        )

        return ClassificationResult(
            pdf_type=PDFType.TEXT_NATIVE,  # arbitrary; confidence=0.0 signals failure
            confidence=0.0,
            tier_used=tier,
            reasoning="LLM classification failed after exhausting retries. Fail-closed.",
            per_page_types={},
            signals=None,
            degraded=False,  # router sets this, not the classifier
        )

    # -- internal helpers ----------------------------------------------------

    def _generate_structural_summary(self, profile: DocumentProfile) -> str:
        """Build a PII-safe structural summary from a DocumentProfile.

        Never includes raw page text -- only structural metadata and
        statistics.

        Args:
            profile: The document profile to summarize.

        Returns:
            A multi-line text summary suitable for inclusion in an LLM prompt.
        """
        filename = os.path.basename(profile.file_path)

        lines: list[str] = []
        lines.append(f"File: {filename}")
        lines.append(f"Pages: {profile.page_count}")
        lines.append(f"File size: {profile.file_size_bytes} bytes")
        lines.append(f"Creator: {profile.metadata.creator or 'unknown'}")
        lines.append(f"PDF version: {profile.metadata.pdf_version or 'unknown'}")
        lines.append("")

        # Page type distribution
        if profile.page_type_distribution:
            lines.append("Page type distribution:")
            for ptype, count in profile.page_type_distribution.items():
                lines.append(f"  - {ptype}: {count}")
            lines.append("")

        # Languages
        if profile.detected_languages:
            lines.append(f"Detected languages: [{', '.join(profile.detected_languages)}]")
        else:
            lines.append("Detected languages: [unknown]")

        # TOC
        if profile.has_toc:
            toc_line = "Table of contents: present"
            if profile.toc_entries:
                toc_line += f" ({len(profile.toc_entries)} entries)"
            lines.append(toc_line)
        else:
            lines.append("Table of contents: not detected")

        # Form fields
        if profile.metadata.has_form_fields:
            lines.append("Form fields: present")
        else:
            lines.append("Form fields: none detected")

        # Encrypted
        if profile.metadata.is_encrypted:
            lines.append("Encrypted: yes")
        else:
            lines.append("Encrypted: no")

        # Security warnings
        if profile.security_warnings:
            lines.append(f"Security warnings: {len(profile.security_warnings)}")

        lines.append("")

        # Sample page profiles
        sample_pages = self._select_sample_pages(profile.pages)
        if sample_pages:
            lines.append("Sample page profiles:")
            for page in sample_pages:
                lines.append(f"Page {page.page_number}:")
                lines.append(
                    f"  - Words: {page.word_count}, Text length: {page.text_length}"
                )
                lines.append(
                    f"  - Images: {page.image_count}, "
                    f"Image coverage: {page.image_coverage_ratio:.1%}"
                )
                lines.append(f"  - Tables: {page.table_count}")
                font_names = ", ".join(page.font_names) if page.font_names else "none"
                lines.append(f"  - Fonts: {page.font_count} ({font_names})")
                lines.append(
                    f"  - Form fields: {'yes' if page.has_form_fields else 'no'}"
                )
                lines.append(
                    f"  - Multi-column: {'yes' if page.is_multi_column else 'no'}"
                )
                lines.append(f"  - Page type: {page.page_type.value}")
                lines.append("")

        return "\n".join(lines)

    def _select_sample_pages(self, pages: list[PageProfile]) -> list[PageProfile]:
        """Select representative sample pages for the structural summary.

        Includes all pages if <= 10, otherwise selects up to 10 pages
        with diversity sampling to cover distinct page types.
        """
        if len(pages) <= 10:
            return pages

        # Diversity sampling: cover all page types, then fill round-robin
        selected: list[PageProfile] = []
        seen_types: set[str] = set()
        remaining: list[PageProfile] = []

        # First pass: one page per distinct type
        for page in pages:
            if page.page_type.value not in seen_types and len(selected) < 10:
                selected.append(page)
                seen_types.add(page.page_type.value)
            else:
                remaining.append(page)

        # Fill remaining slots round-robin
        for page in remaining:
            if len(selected) >= 10:
                break
            selected.append(page)

        # Sort by page number for readability
        selected.sort(key=lambda p: p.page_number)
        return selected

    def _build_classification_prompt(
        self,
        summary: str,
        tier: ClassificationTier,
    ) -> str:
        """Build the classification prompt for the LLM.

        Args:
            summary: The structural summary from ``_generate_structural_summary``.
            tier: The classification tier (available for tier-specific adjustments).

        Returns:
            The full prompt string to send to the LLM backend.
        """
        prompt = (
            "You are classifying a PDF file for a document ingestion system.\n"
            "Based on the structural summary below, classify this file as one of:\n"
            "\n"
            '- "text_native": Digital PDF with extractable text. '
            "Text layer is complete and reliable.\n"
            '- "scanned": Pages are primarily images requiring OCR '
            "for text extraction.\n"
            '- "complex": Mix of text, tables, multi-column layouts, forms, '
            "or combined scanned/digital pages.\n"
            "\n"
            "Respond with JSON only:\n"
            "{\n"
            '  "type": "text_native" | "scanned" | "complex",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "brief explanation",\n'
            '  "page_types": [{"page": 1, "type": "text"}, ...]'
            "  // optional, for complex documents\n"
            "}\n"
            "\n"
            "Valid page types: "
            '"text", "scanned", "table_heavy", "form", "mixed", '
            '"blank", "vector_only", "toc"\n'
            "\n"
            "Structural summary:\n"
            f"{summary}"
        )
        return prompt

    def _validate_and_parse_response(
        self,
        raw: dict,
        profile: DocumentProfile,
    ) -> tuple[LLMClassificationResponse | None, list[IngestError]]:
        """Validate and parse the raw LLM response dict.

        Args:
            raw: The parsed JSON dict from the LLM backend.
            profile: For cross-referencing page numbers in ``page_types``.

        Returns:
            A tuple of (validated response or None, list of errors/warnings).
        """
        errors: list[IngestError] = []

        # Step 1: Pydantic schema validation
        try:
            response = LLMClassificationResponse(**raw)
        except ValidationError as exc:
            errors.append(
                IngestError(
                    code=ErrorCode.E_LLM_SCHEMA_INVALID,
                    message=f"LLM response failed schema validation: {exc}",
                    stage="classify",
                    recoverable=True,
                )
            )
            return None, errors

        # Step 2: Confidence bounds check (manual, since Field has no ge/le)
        if response.confidence < 0.0 or response.confidence > 1.0:
            original = response.confidence
            clamped = max(0.0, min(1.0, response.confidence))
            errors.append(
                IngestError(
                    code=ErrorCode.E_LLM_CONFIDENCE_OOB,
                    message=f"Confidence {original} outside [0.0, 1.0], clamped to {clamped}",
                    stage="classify",
                    recoverable=True,
                )
            )
            response = response.model_copy(update={"confidence": clamped})

        # Step 3: Validate page_types page numbers match profile (if present)
        if response.page_types is not None:
            valid_page_numbers = {p.page_number for p in profile.pages}
            for entry in response.page_types:
                if entry.page not in valid_page_numbers:
                    logger.warning(
                        "LLM returned page_types for unknown page number: %d",
                        entry.page,
                    )

        return response, errors

    def _to_classification_result(
        self,
        response: LLMClassificationResponse,
        tier: ClassificationTier,
    ) -> ClassificationResult:
        """Convert a validated LLM response to a ClassificationResult.

        Args:
            response: The validated LLM response.
            tier: The tier that produced this result.

        Returns:
            A :class:`ClassificationResult` ready for consumption.
        """
        pdf_type = PDFType(response.type)

        # Convert page_types list to dict[int, PageType]
        per_page_types: dict[int, PageType] = {}
        if response.page_types is not None:
            per_page_types = {
                entry.page: PageType(entry.type)
                for entry in response.page_types
            }

        return ClassificationResult(
            pdf_type=pdf_type,
            confidence=response.confidence,
            tier_used=tier,
            reasoning=response.reasoning,
            per_page_types=per_page_types,
            signals=None,  # LLM tiers do not produce signal breakdowns
            degraded=False,  # router sets this, not the classifier
        )

    def _redact(self, text: str) -> str:
        """Apply redaction patterns to text.

        Args:
            text: Text to apply redaction patterns to.

        Returns:
            Redacted text with matching patterns replaced by ``[REDACTED]``.
        """
        result = text
        for pattern in self._config.redact_patterns:
            result = re.sub(pattern, "[REDACTED]", result)
        return result
