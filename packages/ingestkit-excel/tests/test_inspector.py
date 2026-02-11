"""Tests for the ExcelInspector Tier 1 rule-based classifier.

Covers signal evaluation, single-sheet decision logic, multi-sheet
aggregation (agreement / disagreement / hybrid), inconclusive escalation,
edge cases (empty profiles), result field validation, custom configs,
and boundary values.
"""

from __future__ import annotations

import pytest

from ingestkit_excel.config import ExcelProcessorConfig
from ingestkit_excel.inspector import ExcelInspector
from ingestkit_excel.models import (
    ClassificationResult,
    ClassificationTier,
    FileProfile,
    FileType,
    ParserUsed,
    SheetProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sheet_profile(**overrides: object) -> SheetProfile:
    """Build a SheetProfile with sensible tabular defaults."""
    defaults: dict = dict(
        name="Sheet1",
        row_count=100,
        col_count=5,
        merged_cell_count=0,
        merged_cell_ratio=0.0,
        header_row_detected=True,
        header_values=["A", "B", "C", "D", "E"],
        column_type_consistency=0.9,
        numeric_ratio=0.4,
        text_ratio=0.5,
        empty_ratio=0.1,
        sample_rows=[["1", "a", "x", "2.0", "y"]],
        has_formulas=False,
        is_hidden=False,
        parser_used=ParserUsed.OPENPYXL,
    )
    defaults.update(overrides)
    return SheetProfile(**defaults)


def _make_file_profile(
    sheets: list[SheetProfile], **overrides: object
) -> FileProfile:
    """Build a FileProfile from a list of SheetProfiles."""
    defaults: dict = dict(
        file_path="/tmp/test.xlsx",
        file_size_bytes=1024,
        sheet_count=len(sheets),
        sheet_names=[s.name for s in sheets],
        sheets=sheets,
        has_password_protected_sheets=False,
        has_chart_only_sheets=False,
        total_merged_cells=sum(s.merged_cell_count for s in sheets),
        total_rows=sum(s.row_count for s in sheets),
        content_hash="a" * 64,
    )
    defaults.update(overrides)
    return FileProfile(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> ExcelProcessorConfig:
    """Default configuration with standard thresholds."""
    return ExcelProcessorConfig()


@pytest.fixture()
def inspector(config: ExcelProcessorConfig) -> ExcelInspector:
    """Inspector initialised with default config."""
    return ExcelInspector(config)


# ---------------------------------------------------------------------------
# TestSignalEvaluation
# ---------------------------------------------------------------------------


class TestSignalEvaluation:
    """Verify that each of the 5 binary signals is evaluated correctly."""

    def test_row_count_signal_type_a(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(row_count=100)
        signals = inspector._evaluate_signals(sheet)
        assert signals["row_count"] is True

    def test_row_count_signal_type_b(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(row_count=3)
        signals = inspector._evaluate_signals(sheet)
        assert signals["row_count"] is False

    def test_merged_cell_ratio_signal_type_a(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(merged_cell_ratio=0.0)
        signals = inspector._evaluate_signals(sheet)
        assert signals["merged_cell_ratio"] is True

    def test_merged_cell_ratio_signal_type_b(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(merged_cell_ratio=0.1)
        signals = inspector._evaluate_signals(sheet)
        assert signals["merged_cell_ratio"] is False

    def test_column_consistency_signal_type_a(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(column_type_consistency=0.9)
        signals = inspector._evaluate_signals(sheet)
        assert signals["column_type_consistency"] is True

    def test_column_consistency_signal_type_b(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(column_type_consistency=0.4)
        signals = inspector._evaluate_signals(sheet)
        assert signals["column_type_consistency"] is False

    def test_header_detected_signal_type_a(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(header_row_detected=True)
        signals = inspector._evaluate_signals(sheet)
        assert signals["header_detected"] is True

    def test_header_detected_signal_type_b(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(header_row_detected=False)
        signals = inspector._evaluate_signals(sheet)
        assert signals["header_detected"] is False

    def test_numeric_ratio_signal_type_a(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(numeric_ratio=0.5)
        signals = inspector._evaluate_signals(sheet)
        assert signals["numeric_ratio"] is True

    def test_numeric_ratio_signal_type_b(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile(numeric_ratio=0.1)
        signals = inspector._evaluate_signals(sheet)
        assert signals["numeric_ratio"] is False


# ---------------------------------------------------------------------------
# TestSingleSheetClassification
# ---------------------------------------------------------------------------


class TestSingleSheetClassification:
    """Decision logic for a single sheet (via full classify() pipeline)."""

    def test_all_5_signals_type_a_high_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """All 5 signals lean Type A -> tabular_data, confidence 0.9."""
        sheet = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.0,
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.5,
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.9

    def test_all_5_signals_type_b_high_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """All 5 signals lean Type B -> formatted_document, confidence 0.9."""
        sheet = _make_sheet_profile(
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.FORMATTED_DOCUMENT
        assert result.confidence == 0.9

    def test_4_signals_type_a_high_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """4 of 5 signals lean Type A -> tabular_data, confidence 0.9."""
        # numeric_ratio leans Type B; all others lean Type A.
        sheet = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.0,
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.9

    def test_4_signals_type_b_high_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """4 of 5 signals lean Type B -> formatted_document, confidence 0.9."""
        # header_row_detected leans Type A; all others lean Type B.
        sheet = _make_sheet_profile(
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=True,  # Type A
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.FORMATTED_DOCUMENT
        assert result.confidence == 0.9

    def test_3_signals_type_a_medium_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """3 of 5 signals lean Type A -> tabular_data, confidence 0.7."""
        # numeric_ratio and merged_cell_ratio lean Type B; rest lean Type A.
        sheet = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.1,  # Type B
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.7

    def test_3_signals_type_b_medium_confidence(
        self, inspector: ExcelInspector
    ) -> None:
        """3 of 5 signals lean Type B -> formatted_document, confidence 0.7."""
        # row_count and header lean Type A; rest lean Type B.
        sheet = _make_sheet_profile(
            row_count=100,  # Type A
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=True,  # Type A
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.file_type == FileType.FORMATTED_DOCUMENT
        assert result.confidence == 0.7

    def test_inconclusive_with_custom_thresholds(self) -> None:
        """With raised thresholds, 3/5 signals is not enough -> inconclusive."""
        custom_config = ExcelProcessorConfig(
            tier1_high_confidence_signals=5,
            tier1_medium_confidence_signals=4,
        )
        insp = ExcelInspector(custom_config)

        # 3 Type A, 2 Type B -> not enough for medium (4).
        sheet = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.1,  # Type B
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([sheet])
        result = insp.classify(profile)

        assert result.confidence == 0.0
        assert "Inconclusive" in result.reasoning


# ---------------------------------------------------------------------------
# TestMultiSheetAgreement
# ---------------------------------------------------------------------------


class TestMultiSheetAgreement:
    """Multi-sheet logic when all sheets agree on the same type."""

    def test_two_sheets_both_type_a(self, inspector: ExcelInspector) -> None:
        s1 = _make_sheet_profile(name="Data1")
        s2 = _make_sheet_profile(name="Data2")
        profile = _make_file_profile([s1, s2])
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.per_sheet_types is None

    def test_two_sheets_both_type_b(self, inspector: ExcelInspector) -> None:
        kwargs = dict(
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        s1 = _make_sheet_profile(name="Doc1", **kwargs)
        s2 = _make_sheet_profile(name="Doc2", **kwargs)
        profile = _make_file_profile([s1, s2])
        result = inspector.classify(profile)

        assert result.file_type == FileType.FORMATTED_DOCUMENT

    def test_confidence_is_minimum_across_sheets(
        self, inspector: ExcelInspector
    ) -> None:
        """One sheet at 0.9 (5/5), another at 0.7 (3/5) -> file conf 0.7."""
        # Sheet 1: all 5 Type A -> confidence 0.9.
        s1 = _make_sheet_profile(name="HighConf")
        # Sheet 2: 3 Type A -> confidence 0.7.
        s2 = _make_sheet_profile(
            name="MedConf",
            merged_cell_ratio=0.1,  # Type B
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([s1, s2])
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.7

    def test_three_sheets_all_agree(self, inspector: ExcelInspector) -> None:
        sheets = [_make_sheet_profile(name=f"Sheet{i}") for i in range(3)]
        profile = _make_file_profile(sheets)
        result = inspector.classify(profile)

        assert result.file_type == FileType.TABULAR_DATA
        assert result.confidence == 0.9


# ---------------------------------------------------------------------------
# TestMultiSheetDisagreement
# ---------------------------------------------------------------------------


class TestMultiSheetDisagreement:
    """Multi-sheet logic when sheets disagree (hybrid)."""

    def test_two_sheets_disagree_produces_hybrid(
        self, inspector: ExcelInspector
    ) -> None:
        s_tabular = _make_sheet_profile(name="TabSheet")
        s_doc = _make_sheet_profile(
            name="DocSheet",
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([s_tabular, s_doc])
        result = inspector.classify(profile)

        assert result.file_type == FileType.HYBRID
        assert result.confidence == 0.9

    def test_per_sheet_types_populated_on_hybrid(
        self, inspector: ExcelInspector
    ) -> None:
        s_tabular = _make_sheet_profile(name="TabSheet")
        s_doc = _make_sheet_profile(
            name="DocSheet",
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([s_tabular, s_doc])
        result = inspector.classify(profile)

        assert result.per_sheet_types is not None
        assert result.per_sheet_types["TabSheet"] == FileType.TABULAR_DATA
        assert result.per_sheet_types["DocSheet"] == FileType.FORMATTED_DOCUMENT

    def test_three_sheets_two_type_a_one_type_b_is_hybrid(
        self, inspector: ExcelInspector
    ) -> None:
        s1 = _make_sheet_profile(name="Tab1")
        s2 = _make_sheet_profile(name="Tab2")
        s_doc = _make_sheet_profile(
            name="DocSheet",
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        profile = _make_file_profile([s1, s2, s_doc])
        result = inspector.classify(profile)

        assert result.file_type == FileType.HYBRID
        assert result.per_sheet_types is not None
        assert len(result.per_sheet_types) == 3


# ---------------------------------------------------------------------------
# TestInconclusiveEscalation
# ---------------------------------------------------------------------------


class TestInconclusiveEscalation:
    """Inconclusive results should have low confidence for tier escalation."""

    def test_inconclusive_sheet_produces_low_confidence(self) -> None:
        """A single inconclusive sheet -> confidence 0.0."""
        custom_config = ExcelProcessorConfig(
            tier1_high_confidence_signals=5,
            tier1_medium_confidence_signals=4,
        )
        insp = ExcelInspector(custom_config)

        # 3 Type A, 2 Type B -> not enough for medium (4).
        sheet = _make_sheet_profile(
            merged_cell_ratio=0.1,  # Type B
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([sheet])
        result = insp.classify(profile)

        assert result.confidence == 0.0

    def test_inconclusive_sheet_among_clear_sheets(self) -> None:
        """If one sheet is inconclusive among clear ones, whole file is inconclusive."""
        custom_config = ExcelProcessorConfig(
            tier1_high_confidence_signals=5,
            tier1_medium_confidence_signals=4,
        )
        insp = ExcelInspector(custom_config)

        # Sheet 1: all 5 Type A -> clear.
        s1 = _make_sheet_profile(name="ClearSheet")
        # Sheet 2: 3 Type A, 2 Type B -> inconclusive under custom config.
        s2 = _make_sheet_profile(
            name="AmbiguousSheet",
            merged_cell_ratio=0.1,  # Type B
            numeric_ratio=0.1,  # Type B
        )
        profile = _make_file_profile([s1, s2])
        result = insp.classify(profile)

        assert result.confidence == 0.0
        assert "Inconclusive" in result.reasoning
        assert "AmbiguousSheet" in result.reasoning


# ---------------------------------------------------------------------------
# TestEmptyProfile
# ---------------------------------------------------------------------------


class TestEmptyProfile:
    """Edge cases with empty/no sheets."""

    def test_empty_sheets_list_returns_inconclusive(
        self, inspector: ExcelInspector
    ) -> None:
        profile = _make_file_profile([], sheet_count=0)
        result = inspector.classify(profile)

        assert result.confidence == 0.0
        assert "no sheets" in result.reasoning.lower()

    def test_zero_sheet_count_returns_inconclusive(
        self, inspector: ExcelInspector
    ) -> None:
        profile = _make_file_profile([], sheet_count=0)
        result = inspector.classify(profile)

        assert result.confidence == 0.0
        assert result.tier_used == ClassificationTier.RULE_BASED


# ---------------------------------------------------------------------------
# TestClassificationResultFields
# ---------------------------------------------------------------------------


class TestClassificationResultFields:
    """Verify output fields are correctly populated."""

    def test_tier_used_is_always_rule_based(
        self, inspector: ExcelInspector
    ) -> None:
        sheet = _make_sheet_profile()
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.tier_used == ClassificationTier.RULE_BASED
        assert result.tier_used.value == "rule_based"

    def test_signals_dict_populated(self, inspector: ExcelInspector) -> None:
        sheet = _make_sheet_profile()
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert result.signals is not None
        assert "per_sheet" in result.signals
        assert "Sheet1" in result.signals["per_sheet"]

        sheet_signals = result.signals["per_sheet"]["Sheet1"]["signals"]
        assert "row_count" in sheet_signals
        assert "merged_cell_ratio" in sheet_signals
        assert "column_type_consistency" in sheet_signals
        assert "header_detected" in sheet_signals
        assert "numeric_ratio" in sheet_signals

    def test_reasoning_is_non_empty_string(
        self, inspector: ExcelInspector
    ) -> None:
        sheet = _make_sheet_profile()
        profile = _make_file_profile([sheet])
        result = inspector.classify(profile)

        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_file_type_uses_enum_values(
        self, inspector: ExcelInspector
    ) -> None:
        # Type A file.
        sheet_a = _make_sheet_profile()
        result_a = inspector.classify(_make_file_profile([sheet_a]))
        assert result_a.file_type.value == "tabular_data"

        # Type B file.
        sheet_b = _make_sheet_profile(
            row_count=3,
            merged_cell_ratio=0.2,
            column_type_consistency=0.3,
            header_row_detected=False,
            numeric_ratio=0.1,
        )
        result_b = inspector.classify(_make_file_profile([sheet_b]))
        assert result_b.file_type.value == "formatted_document"

        # Hybrid file.
        result_h = inspector.classify(_make_file_profile([sheet_a, sheet_b]))
        assert result_h.file_type.value == "hybrid"


# ---------------------------------------------------------------------------
# TestCustomConfig
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify that custom thresholds are respected."""

    def test_custom_min_row_count_threshold(self) -> None:
        """min_row_count_for_tabular=50, row_count=30 -> signal leans Type B."""
        cfg = ExcelProcessorConfig(min_row_count_for_tabular=50)
        insp = ExcelInspector(cfg)
        sheet = _make_sheet_profile(row_count=30)
        signals = insp._evaluate_signals(sheet)

        assert signals["row_count"] is False

    def test_custom_merged_cell_ratio_threshold(self) -> None:
        """merged_cell_ratio_threshold=0.2, ratio=0.1 -> still Type A."""
        cfg = ExcelProcessorConfig(merged_cell_ratio_threshold=0.2)
        insp = ExcelInspector(cfg)
        sheet = _make_sheet_profile(merged_cell_ratio=0.1)
        signals = insp._evaluate_signals(sheet)

        assert signals["merged_cell_ratio"] is True

    def test_custom_confidence_signal_counts(self) -> None:
        """tier1_high=5, tier1_medium=4 -> 4/5 is medium, 3/5 is inconclusive."""
        cfg = ExcelProcessorConfig(
            tier1_high_confidence_signals=5,
            tier1_medium_confidence_signals=4,
        )
        insp = ExcelInspector(cfg)

        # 4 Type A signals -> medium confidence.
        sheet_4a = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.0,
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.1,  # Type B
        )
        result_4a = insp.classify(_make_file_profile([sheet_4a]))
        assert result_4a.file_type == FileType.TABULAR_DATA
        assert result_4a.confidence == 0.7

        # 5 Type A signals -> high confidence.
        sheet_5a = _make_sheet_profile()
        result_5a = insp.classify(_make_file_profile([sheet_5a]))
        assert result_5a.file_type == FileType.TABULAR_DATA
        assert result_5a.confidence == 0.9

        # 3 Type A signals -> inconclusive.
        sheet_3a = _make_sheet_profile(
            row_count=100,
            merged_cell_ratio=0.1,  # Type B
            column_type_consistency=0.9,
            header_row_detected=True,
            numeric_ratio=0.1,  # Type B
        )
        result_3a = insp.classify(_make_file_profile([sheet_3a]))
        assert result_3a.confidence == 0.0


# ---------------------------------------------------------------------------
# TestBoundaryValues
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    """Test exact boundary conditions."""

    def test_row_count_exactly_at_threshold(
        self, inspector: ExcelInspector
    ) -> None:
        """row_count=5 (== min_row_count_for_tabular=5) -> Type A signal."""
        sheet = _make_sheet_profile(row_count=5)
        signals = inspector._evaluate_signals(sheet)
        assert signals["row_count"] is True

    def test_merged_ratio_exactly_at_threshold(
        self, inspector: ExcelInspector
    ) -> None:
        """merged_cell_ratio=0.05 (== threshold) -> Type B signal (>= threshold)."""
        sheet = _make_sheet_profile(merged_cell_ratio=0.05)
        signals = inspector._evaluate_signals(sheet)
        assert signals["merged_cell_ratio"] is False

    def test_column_consistency_exactly_at_threshold(
        self, inspector: ExcelInspector
    ) -> None:
        """column_type_consistency=0.7 (== threshold) -> Type A signal (>= threshold)."""
        sheet = _make_sheet_profile(column_type_consistency=0.7)
        signals = inspector._evaluate_signals(sheet)
        assert signals["column_type_consistency"] is True

    def test_numeric_ratio_exactly_at_threshold(
        self, inspector: ExcelInspector
    ) -> None:
        """numeric_ratio=0.3 (== threshold) -> Type A signal (>= threshold)."""
        sheet = _make_sheet_profile(numeric_ratio=0.3)
        signals = inspector._evaluate_signals(sheet)
        assert signals["numeric_ratio"] is True
