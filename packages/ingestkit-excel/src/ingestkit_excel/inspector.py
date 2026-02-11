"""Tier 1 rule-based structural inspector for Excel files.

Evaluates five binary signals per sheet and applies threshold-based decision
logic to classify files as tabular data (Type A), formatted document (Type B),
or hybrid (Type C) -- without any LLM call.

Signal evaluation and multi-sheet aggregation logic are defined in SPEC.md
section 8.  All configurable thresholds live in
:class:`~ingestkit_excel.config.ExcelProcessorConfig`.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationTier,
    FileProfile,
    FileType,
    SheetProfile,
)

logger = logging.getLogger("ingestkit_excel")


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


class ExcelInspector:
    """Tier 1 rule-based structural inspector for Excel files.

    Evaluates 5 binary signals per sheet and uses threshold-based
    decision logic to classify files without any LLM call.
    """

    def __init__(self, config: ExcelProcessorConfig) -> None:
        self._config = config

    # -- public API ----------------------------------------------------------

    def classify(self, profile: FileProfile) -> ClassificationResult:
        """Classify a file based on structural signals from its profile.

        Args:
            profile: The :class:`FileProfile` produced by the parser chain.

        Returns:
            A :class:`ClassificationResult` with file type, confidence,
            tier information, and signal breakdown.
        """
        # Edge case: no sheets at all.
        if profile.sheet_count == 0 or len(profile.sheets) == 0:
            logger.info(
                "No sheets found in %s -- returning inconclusive.", profile.file_path
            )
            return ClassificationResult(
                file_type=FileType.HYBRID,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning="Inconclusive: file contains no sheets to classify.",
                signals={"per_sheet": {}},
            )

        # Classify each sheet independently.
        sheet_results: list[tuple[str, FileType | None, float, dict[str, bool]]] = []
        for sheet in profile.sheets:
            file_type, confidence, signals = self._classify_sheet(sheet)
            sheet_results.append((sheet.name, file_type, confidence, signals))

        # Build aggregated signals dict.
        signals: dict[str, Any] = {
            "per_sheet": {
                name: {
                    "type": ft.value if ft else None,
                    "confidence": conf,
                    "signals": sigs,
                }
                for name, ft, conf, sigs in sheet_results
            }
        }

        # Check for any inconclusive sheet.
        inconclusive_sheets = [
            name for name, ft, _conf, _sigs in sheet_results if ft is None
        ]
        if inconclusive_sheets:
            first = inconclusive_sheets[0]
            reasoning = (
                f"Inconclusive: sheet '{first}' could not be classified "
                "with sufficient confidence."
            )
            logger.info(
                "Inconclusive classification for %s -- %s",
                profile.file_path,
                reasoning,
            )
            return ClassificationResult(
                file_type=FileType.HYBRID,
                confidence=0.0,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning=reasoning,
                signals=signals,
            )

        # All sheets classified -- check agreement.
        # At this point every file_type is not None.
        distinct_types = {ft for _name, ft, _conf, _sigs in sheet_results}

        if len(distinct_types) == 1:
            # All sheets agree.
            agreed_type: FileType = next(iter(distinct_types))  # type: ignore[assignment]
            min_confidence = min(conf for _name, _ft, conf, _sigs in sheet_results)
            n = len(sheet_results)
            reasoning = (
                f"All {n} sheet(s) classified as {agreed_type.value} with "
                f"{min_confidence} confidence by Tier 1 rule-based inspector."
            )
            logger.info(
                "Classified %s as %s (confidence=%s, tier=rule_based).",
                profile.file_path,
                agreed_type.value,
                min_confidence,
            )
            return ClassificationResult(
                file_type=agreed_type,
                confidence=min_confidence,
                tier_used=ClassificationTier.RULE_BASED,
                reasoning=reasoning,
                signals=signals,
            )

        # Sheets disagree -- hybrid.
        per_sheet_types: dict[str, FileType] = {
            name: ft  # type: ignore[misc]
            for name, ft, _conf, _sigs in sheet_results
        }

        type_a_names = [
            name
            for name, ft, _conf, _sigs in sheet_results
            if ft == FileType.TABULAR_DATA
        ]
        type_b_names = [
            name
            for name, ft, _conf, _sigs in sheet_results
            if ft == FileType.FORMATTED_DOCUMENT
        ]

        reasoning = (
            f"Sheets disagree: {type_a_names} are tabular_data, "
            f"{type_b_names} are formatted_document. Classified as hybrid."
        )
        logger.info(
            "Classified %s as hybrid (tier=rule_based).", profile.file_path
        )
        return ClassificationResult(
            file_type=FileType.HYBRID,
            confidence=0.9,
            tier_used=ClassificationTier.RULE_BASED,
            reasoning=reasoning,
            per_sheet_types=per_sheet_types,
            signals=signals,
        )

    # -- internal helpers ----------------------------------------------------

    def _classify_sheet(
        self, sheet: SheetProfile
    ) -> tuple[FileType | None, float, dict[str, bool]]:
        """Classify a single sheet.

        Returns:
            A tuple of ``(file_type_or_None, confidence, signals_dict)``.
            ``file_type`` is ``None`` when the result is inconclusive.
        """
        signals = self._evaluate_signals(sheet)
        type_a_count = sum(1 for v in signals.values() if v)
        type_b_count = 5 - type_a_count

        logger.debug(
            "Sheet '%s': type_a=%d, type_b=%d, signals=%s",
            sheet.name,
            type_a_count,
            type_b_count,
            signals,
        )

        cfg = self._config
        if type_a_count >= cfg.tier1_high_confidence_signals:
            return FileType.TABULAR_DATA, 0.9, signals
        if type_b_count >= cfg.tier1_high_confidence_signals:
            return FileType.FORMATTED_DOCUMENT, 0.9, signals
        if type_a_count >= cfg.tier1_medium_confidence_signals:
            return FileType.TABULAR_DATA, 0.7, signals
        if type_b_count >= cfg.tier1_medium_confidence_signals:
            return FileType.FORMATTED_DOCUMENT, 0.7, signals

        # Inconclusive.
        return None, 0.0, signals

    def _evaluate_signals(self, sheet: SheetProfile) -> dict[str, bool]:
        """Evaluate the 5 binary signals for a sheet.

        Each signal maps to ``True`` if the sheet leans Type A (tabular)
        and ``False`` if it leans Type B (formatted document).
        """
        cfg = self._config
        return {
            "row_count": sheet.row_count >= cfg.min_row_count_for_tabular,
            "merged_cell_ratio": sheet.merged_cell_ratio < cfg.merged_cell_ratio_threshold,
            "column_type_consistency": sheet.column_type_consistency >= cfg.column_consistency_threshold,
            "header_detected": sheet.header_row_detected,
            "numeric_ratio": sheet.numeric_ratio >= cfg.numeric_ratio_threshold,
        }
