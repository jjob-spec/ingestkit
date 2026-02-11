"""Tier 2/3 LLM-based file classifier with schema validation.

Generates a structural summary from a FileProfile, sends it to an LLM
backend via a classification prompt, validates the response against a
Pydantic schema, and returns a ClassificationResult.

The classifier does NOT handle tier escalation logic -- that is the
responsibility of the router.  It classifies at the specific tier
requested.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.errors import ErrorCode, IngestError
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationTier,
    FileProfile,
    FileType,
    SheetProfile,
)
from ingestkit_excel.protocols import LLMBackend

logger = logging.getLogger("ingestkit_excel")


# ---------------------------------------------------------------------------
# LLM Response Schema
# ---------------------------------------------------------------------------


class LLMClassificationResponse(BaseModel):
    """Schema for validating LLM classification output.

    The ``type`` field uses Literal string values matching the ``FileType``
    enum values.  Confidence bounds are checked manually (not via Field
    constraints) to allow clamping instead of rejection.
    """

    type: Literal["tabular_data", "formatted_document", "hybrid"]
    confidence: float  # NO ge/le constraints -- checked manually to allow clamping
    reasoning: str = Field(min_length=1)
    sheet_types: dict[str, Literal["tabular_data", "formatted_document"]] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_cell_type(value: str) -> str:
    """Infer the structural type of a cell value for the LLM summary."""
    if not value or value.strip() == "":
        return "empty"
    try:
        int(value)
        return "int"
    except ValueError:
        pass
    try:
        float(value)
        return "float"
    except ValueError:
        pass
    return "str"


# ---------------------------------------------------------------------------
# LLM Classifier
# ---------------------------------------------------------------------------


class LLMClassifier:
    """Tier 2/3 LLM-based file classifier with schema validation.

    Generates a structural summary from a FileProfile, sends it to an LLM
    backend via a classification prompt, validates the response against a
    Pydantic schema, and returns a ClassificationResult.

    The classifier does NOT handle tier escalation logic -- that is the
    responsibility of the router.  It classifies at the specific tier
    requested.
    """

    def __init__(self, llm: LLMBackend, config: ExcelProcessorConfig) -> None:
        self._llm = llm
        self._config = config

    # -- public API ----------------------------------------------------------

    def classify(
        self,
        profile: FileProfile,
        tier: ClassificationTier,
    ) -> ClassificationResult:
        """Classify a file at the specified LLM tier.

        Args:
            profile: The structural profile of the Excel file.
            tier: Which tier to run (``LLM_BASIC`` or ``LLM_REASONING``).

        Returns:
            A :class:`ClassificationResult` with file type, confidence,
            tier information, and reasoning.

        Raises:
            ValueError: If ``tier`` is ``ClassificationTier.RULE_BASED``.
        """
        # 1. Validate tier parameter
        if tier == ClassificationTier.RULE_BASED:
            raise ValueError("LLMClassifier does not handle rule_based tier")

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
                # Record retry warning
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
                # Backend failed to parse JSON
                errors.append(
                    IngestError(
                        code=ErrorCode.E_LLM_MALFORMED_JSON,
                        message=f"LLM returned unparseable JSON: {exc}",
                        stage="classify",
                        recoverable=True,
                    )
                )
                # Append correction hint to prompt for retry
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
                    'Ensure \'type\' is one of "tabular_data", "formatted_document", '
                    '"hybrid". Ensure \'confidence\' is a float. Ensure \'reasoning\' '
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

        # Return an inconclusive result with zero confidence
        return ClassificationResult(
            file_type=FileType.TABULAR_DATA,  # arbitrary; confidence=0.0 signals failure
            confidence=0.0,
            tier_used=tier,
            reasoning="LLM classification failed after exhausting retries. Fail-closed.",
            per_sheet_types=None,
            signals=None,
        )

    # -- internal helpers ----------------------------------------------------

    def _generate_structural_summary(self, profile: FileProfile) -> str:
        """Build a PII-safe structural summary from a FileProfile.

        Args:
            profile: The file profile to summarize.

        Returns:
            A multi-line text summary suitable for inclusion in an LLM prompt.
        """
        # Extract filename only from file_path (no filesystem info leakage)
        filename = os.path.basename(profile.file_path)

        lines: list[str] = []
        lines.append(f"File: {filename}")
        lines.append(
            f"Sheets: {profile.sheet_count} ({', '.join(profile.sheet_names)})"
        )

        if profile.has_password_protected_sheets:
            lines.append("Note: File contains password-protected sheets.")
        if profile.has_chart_only_sheets:
            lines.append("Note: File contains chart-only sheets.")

        lines.append("")  # blank line

        for sheet in profile.sheets:
            lines.append(f'Sheet "{sheet.name}":')
            lines.append(f"- Rows: {sheet.row_count}, Columns: {sheet.col_count}")
            lines.append(
                f"- Merged cells: {sheet.merged_cell_count} "
                f"(ratio: {sheet.merged_cell_ratio:.3f})"
            )

            if sheet.header_values:
                lines.append(f"- Headers: [{', '.join(sheet.header_values)}]")
            else:
                lines.append("- Headers: [none detected]")

            if sheet.has_formulas:
                lines.append("- Contains formulas: yes")
            if sheet.is_hidden:
                lines.append("- Hidden sheet: yes")

            # Sample rows section
            max_rows = min(self._config.max_sample_rows, len(sheet.sample_rows))
            if max_rows > 0:
                if self._config.log_sample_data:
                    # Include actual values (with redaction)
                    lines.append("- Sample rows (values):")
                    for i, row in enumerate(sheet.sample_rows[:max_rows]):
                        redacted_row = [self._redact(cell) for cell in row]
                        lines.append(f"  Row {i + 1}: [{', '.join(redacted_row)}]")
                else:
                    # Structure-only: show types, never raw values
                    lines.append("- Sample rows (structure only):")
                    for i, row in enumerate(sheet.sample_rows[:max_rows]):
                        types = [_infer_cell_type(cell) for cell in row]
                        lines.append(f"  Row {i + 1}: [{', '.join(types)}]")

            lines.append("")  # blank line between sheets

        return "\n".join(lines)

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
            "You are classifying an Excel file for a document ingestion system.\n"
            "Based on the structural summary below, classify this file as one of:\n"
            "\n"
            '- "tabular_data": Rows are records, columns are fields. '
            "Consistent structure. Suitable for SQL database import.\n"
            '- "formatted_document": Excel used as a layout/formatting tool. '
            "Merged cells, irregular structure, text-heavy. "
            "Suitable for text extraction.\n"
            '- "hybrid": Mix of tabular and document-formatted sections. '
            "Different sheets or regions serve different purposes.\n"
            "\n"
            "Respond with JSON only:\n"
            "{\n"
            '  "type": "tabular_data" | "formatted_document" | "hybrid",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "brief explanation",\n'
            '  "sheet_types": {"sheet_name": "type", ...}  // only if hybrid\n'
            "}\n"
            "\n"
            "Structural summary:\n"
            f"{summary}"
        )
        return prompt

    def _validate_and_parse_response(
        self,
        raw: dict,
        profile: FileProfile,
    ) -> tuple[LLMClassificationResponse | None, list[IngestError]]:
        """Validate and parse the raw LLM response dict.

        Args:
            raw: The parsed JSON dict from the LLM backend.
            profile: For cross-referencing sheet names in ``sheet_types``.

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
            # Create new response with clamped confidence
            response = response.model_copy(update={"confidence": clamped})

        # Step 3: Validate sheet_types keys match actual sheet names (if hybrid)
        if response.type == "hybrid" and response.sheet_types is not None:
            known_sheets = set(profile.sheet_names)
            unknown_sheets = set(response.sheet_types.keys()) - known_sheets
            if unknown_sheets:
                # Warning only -- do not reject
                logger.warning(
                    "LLM returned sheet_types for unknown sheets: %s", unknown_sheets
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
        # Convert string type to FileType enum using VALUE form
        file_type = FileType(response.type)

        # Convert sheet_types strings to FileType enums (if present)
        per_sheet_types: dict[str, FileType] | None = None
        if response.sheet_types is not None:
            per_sheet_types = {
                name: FileType(st) for name, st in response.sheet_types.items()
            }

        return ClassificationResult(
            file_type=file_type,
            confidence=response.confidence,
            tier_used=tier,
            reasoning=response.reasoning,
            per_sheet_types=per_sheet_types,
            signals=None,  # LLM tiers do not produce signal breakdowns
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
