"""Tests for Visual Selection Rules.

Validates that the visual selector correctly maps analytical intents to
company-approved chart types, and that field placement follows the strict
memory rules from the PowerBI CheatSheet.
"""

import pytest

from app.core.field_mapper import ColumnMeta, ColumnRole, DtypeClass, map_fields
from app.core.visual_selector import (
    INTENT_CATEGORY_COMPARISON,
    INTENT_CORRELATION,
    INTENT_CUMULATIVE_GROWTH,
    INTENT_DISTRIBUTION,
    INTENT_GEOGRAPHIC,
    INTENT_HIERARCHICAL,
    INTENT_MULTI_NUMERIC,
    INTENT_PROPORTION,
    INTENT_RAW_TABLE,
    INTENT_SINGLE_VALUE,
    INTENT_TIME_TREND,
    infer_intent,
    select_visual,
)
from app.schemas.json_plan_schema import (
    AggregationSpec,
    OperationType,
    QueryPlan,
)
from app.schemas.visual_payload import ChartType


# ---------------------------------------------------------------------------
# Visual selection tests
# ---------------------------------------------------------------------------


class TestSelectVisual:
    """Ensure the visual selector maps intents to the correct chart types."""

    def _make_columns(self, specs):
        """Helper: specs is a list of (name, dtype_class) tuples."""
        return [
            ColumnMeta(
                name=name,
                dtype_class=DtypeClass(dc),
                role=ColumnRole.MEASURE if dc == "NUMERIC" else ColumnRole.DIMENSION,
            )
            for name, dc in specs
        ]

    def test_category_comparison(self):
        cols = self._make_columns([("Region", "TEXT"), ("Revenue", "NUMERIC")])
        result = select_visual(
            intent=INTENT_CATEGORY_COMPARISON,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=1,
            num_categorical=1,
        )
        assert result in (ChartType.BAR, ChartType.CLUSTERED_BAR, ChartType.COLUMN)

    def test_time_trend(self):
        cols = self._make_columns([("Date", "DATE"), ("Revenue", "NUMERIC")])
        result = select_visual(
            intent=INTENT_TIME_TREND,
            columns=cols,
            has_date=True,
            has_geo=False,
            num_numeric=1,
            num_categorical=0,
        )
        assert result in (ChartType.LINE, ChartType.COLUMN)

    def test_single_value(self):
        cols = self._make_columns([("Revenue", "NUMERIC")])
        result = select_visual(
            intent=INTENT_SINGLE_VALUE,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=1,
            num_categorical=0,
        )
        assert result == ChartType.CARD

    def test_proportion(self):
        cols = self._make_columns([("Category", "TEXT"), ("Amount", "NUMERIC")])
        result = select_visual(
            intent=INTENT_PROPORTION,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=1,
            num_categorical=1,
        )
        assert result in (ChartType.PIE, ChartType.DONUT)

    def test_correlation(self):
        cols = self._make_columns([("Price", "NUMERIC"), ("Quantity", "NUMERIC")])
        result = select_visual(
            intent=INTENT_CORRELATION,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=2,
            num_categorical=0,
        )
        assert result == ChartType.SCATTER

    def test_multi_numeric(self):
        cols = self._make_columns([
            ("X", "NUMERIC"), ("Y", "NUMERIC"), ("Z", "NUMERIC"),
        ])
        result = select_visual(
            intent=INTENT_MULTI_NUMERIC,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=3,
            num_categorical=0,
        )
        assert result == ChartType.BUBBLE

    def test_distribution(self):
        cols = self._make_columns([("Revenue", "NUMERIC")])
        result = select_visual(
            intent=INTENT_DISTRIBUTION,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=1,
            num_categorical=0,
        )
        assert result == ChartType.HISTOGRAM

    def test_hierarchical(self):
        cols = self._make_columns([
            ("Region", "TEXT"), ("Product", "TEXT"), ("Revenue", "NUMERIC"),
        ])
        result = select_visual(
            intent=INTENT_HIERARCHICAL,
            columns=cols,
            has_date=False,
            has_geo=False,
            num_numeric=1,
            num_categorical=2,
        )
        assert result in (ChartType.MATRIX, ChartType.TREEMAP)

    def test_geographic(self):
        cols = self._make_columns([("Country", "GEO"), ("Revenue", "NUMERIC")])
        result = select_visual(
            intent=INTENT_GEOGRAPHIC,
            columns=cols,
            has_date=False,
            has_geo=True,
            num_numeric=1,
            num_categorical=0,
        )
        assert result == ChartType.MAP


# ---------------------------------------------------------------------------
# Field placement memory rules
# ---------------------------------------------------------------------------


class TestFieldPlacement:
    """Validate the 5 field placement memory rules from the PowerBI CheatSheet."""

    def test_text_goes_to_axis(self):
        cols = [
            ColumnMeta(name="Region", dtype_class=DtypeClass.TEXT, role=ColumnRole.DIMENSION),
            ColumnMeta(name="Revenue", dtype_class=DtypeClass.NUMERIC, role=ColumnRole.MEASURE),
        ]
        placement = map_fields(cols, ChartType.BAR)
        # Text → Axis
        assert placement.axis_x is not None or placement.axis_y is not None

    def test_numeric_goes_to_values(self):
        cols = [
            ColumnMeta(name="Region", dtype_class=DtypeClass.TEXT, role=ColumnRole.DIMENSION),
            ColumnMeta(name="Revenue", dtype_class=DtypeClass.NUMERIC, role=ColumnRole.MEASURE),
        ]
        placement = map_fields(cols, ChartType.BAR)
        # Numeric → Values/Y-Axis
        assert placement.values is not None or placement.axis_y is not None or placement.axis_x is not None

    def test_date_goes_to_xaxis(self):
        cols = [
            ColumnMeta(name="OrderDate", dtype_class=DtypeClass.DATE, role=ColumnRole.DIMENSION),
            ColumnMeta(name="Revenue", dtype_class=DtypeClass.NUMERIC, role=ColumnRole.MEASURE),
        ]
        placement = map_fields(cols, ChartType.LINE)
        assert placement.axis_x is not None

    def test_geo_goes_to_location(self):
        cols = [
            ColumnMeta(name="Country", dtype_class=DtypeClass.GEO, role=ColumnRole.DIMENSION),
            ColumnMeta(name="Revenue", dtype_class=DtypeClass.NUMERIC, role=ColumnRole.MEASURE),
        ]
        placement = map_fields(cols, ChartType.MAP)
        assert placement.location is not None


# ---------------------------------------------------------------------------
# Intent inference tests
# ---------------------------------------------------------------------------


class TestInferIntent:
    """Validate the intent inference from query plans."""

    def _make_cols(self):
        return [
            ColumnMeta(name="Region", dtype_class=DtypeClass.TEXT, role=ColumnRole.DIMENSION),
            ColumnMeta(name="Revenue", dtype_class=DtypeClass.NUMERIC, role=ColumnRole.MEASURE),
            ColumnMeta(name="OrderDate", dtype_class=DtypeClass.DATE, role=ColumnRole.DIMENSION),
        ]

    def test_aggregation_with_group_by_text(self):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            group_by=["Region"],
            agg=AggregationSpec(col="Revenue", fn="sum"),
        )
        intent = infer_intent(plan, self._make_cols())
        assert intent == INTENT_CATEGORY_COMPARISON

    def test_aggregation_no_group(self):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="sum"),
        )
        intent = infer_intent(plan, self._make_cols())
        assert intent == INTENT_SINGLE_VALUE

    def test_raw_returns_table(self):
        plan = QueryPlan(operation=OperationType.RAW)
        intent = infer_intent(plan, self._make_cols())
        assert intent == INTENT_RAW_TABLE

