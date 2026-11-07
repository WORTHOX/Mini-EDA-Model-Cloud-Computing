"""Tests for the Automated Baseline EDA Engine (Page 1).

Validates that CSV upload triggers correct Univariate, Bivariate, and
Multivariate chart generation across different data shapes.
"""

import pandas as pd
import pytest

from app.core.page1_baseline_eda import (
    BaselineEDAResult,
    ColumnMeta,
    classify_columns,
    generate_bivariate,
    generate_multivariate,
    generate_univariate,
    run_baseline_eda,
)
from app.schemas.visual_payload import ChartType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small mixed-type DataFrame for testing."""
    return pd.DataFrame(
        {
            "OrderDate": pd.date_range("2024-01-01", periods=50, freq="D"),
            "Region": ["North", "South", "East", "West", "Central"] * 10,
            "Product": [f"Product_{i % 8}" for i in range(50)],
            "Revenue": [round(100 + i * 3.5, 2) for i in range(50)],
            "Quantity": [10 + i % 20 for i in range(50)],
            "Profit": [round(20 + i * 1.1, 2) for i in range(50)],
        }
    )


@pytest.fixture
def numeric_only_df() -> pd.DataFrame:
    """A DataFrame with only numeric columns."""
    return pd.DataFrame(
        {
            "MetricA": range(30),
            "MetricB": [x * 2.5 for x in range(30)],
            "MetricC": [x ** 0.5 for x in range(30)],
        }
    )


@pytest.fixture
def categorical_only_df() -> pd.DataFrame:
    """A DataFrame with only categorical columns."""
    return pd.DataFrame(
        {
            "Status": ["Active", "Inactive", "Pending"] * 10,
            "Department": ["Sales", "Engineering", "HR", "Marketing", "Finance"] * 6,
        }
    )


# ---------------------------------------------------------------------------
# Column classification tests
# ---------------------------------------------------------------------------


class TestClassifyColumns:
    def test_detects_numeric_columns(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        numeric_names = {c.name for c in cols if c.dtype_class == "NUMERIC"}
        assert "Revenue" in numeric_names
        assert "Quantity" in numeric_names
        assert "Profit" in numeric_names

    def test_detects_text_columns(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        text_names = {c.name for c in cols if c.dtype_class == "TEXT"}
        assert "Product" in text_names
        # "Region" is classified as GEO (matches geo keyword), which is correct
        geo_names = {c.name for c in cols if c.dtype_class == "GEO"}
        assert "Region" in text_names or "Region" in geo_names

    def test_detects_date_columns(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        date_names = {c.name for c in cols if c.dtype_class == "DATE"}
        assert "OrderDate" in date_names

    def test_all_columns_classified(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        assert len(cols) == len(sample_df.columns)


# ---------------------------------------------------------------------------
# Univariate analysis tests
# ---------------------------------------------------------------------------


class TestUnivariate:
    def test_generates_kpi_cards_for_numerics(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_univariate(sample_df, cols)
        card_payloads = [p for p in payloads if p.chart_type == ChartType.CARD]
        assert len(card_payloads) > 0, "Should generate KPI cards for numeric columns"

    def test_generates_histograms(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_univariate(sample_df, cols)
        hist_payloads = [p for p in payloads if p.chart_type == ChartType.HISTOGRAM]
        assert len(hist_payloads) > 0, "Should generate histograms for numeric columns"

    def test_generates_bar_charts_for_categoricals(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_univariate(sample_df, cols)
        bar_payloads = [
            p
            for p in payloads
            if p.chart_type in (ChartType.BAR, ChartType.COLUMN, ChartType.CLUSTERED_BAR)
        ]
        assert len(bar_payloads) > 0, "Should generate bar charts for categorical columns"

    def test_empty_for_empty_df(self):
        df = pd.DataFrame()
        cols = classify_columns(df)
        payloads = generate_univariate(df, cols)
        assert payloads == []


# ---------------------------------------------------------------------------
# Bivariate analysis tests
# ---------------------------------------------------------------------------


class TestBivariate:
    def test_generates_time_series(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_bivariate(sample_df, cols)
        line_payloads = [
            p for p in payloads if p.chart_type in (ChartType.LINE, ChartType.COLUMN)
        ]
        assert len(line_payloads) > 0, "Should generate time-series charts for Date×Numeric pairs"

    def test_generates_categorical_bars(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_bivariate(sample_df, cols)
        bar_payloads = [
            p
            for p in payloads
            if p.chart_type in (ChartType.CLUSTERED_BAR, ChartType.BAR, ChartType.CLUSTERED_COLUMN)
        ]
        assert len(bar_payloads) > 0, "Should generate categorical comparison bar charts"

    def test_generates_scatter_plots(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_bivariate(sample_df, cols)
        scatter_payloads = [p for p in payloads if p.chart_type == ChartType.SCATTER]
        assert len(scatter_payloads) > 0, "Should generate scatter plots for Numeric×Numeric pairs"


# ---------------------------------------------------------------------------
# Multivariate analysis tests
# ---------------------------------------------------------------------------


class TestMultivariate:
    def test_generates_bubble_chart(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_multivariate(sample_df, cols)
        bubble_payloads = [p for p in payloads if p.chart_type == ChartType.BUBBLE]
        assert len(bubble_payloads) > 0, "Should generate bubble charts when 3+ numerics exist"

    def test_generates_matrix(self, sample_df: pd.DataFrame):
        cols = classify_columns(sample_df)
        payloads = generate_multivariate(sample_df, cols)
        matrix_payloads = [p for p in payloads if p.chart_type == ChartType.MATRIX]
        assert len(matrix_payloads) > 0, "Should generate pivot/matrix for 2+ text + 1+ numeric"


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestRunBaselineEDA:
    def test_returns_all_tiers(self, sample_df: pd.DataFrame):
        result = run_baseline_eda(sample_df)
        assert isinstance(result, BaselineEDAResult)
        assert len(result.univariate) > 0
        assert len(result.bivariate) > 0
        assert len(result.multivariate) > 0
        assert result.row_count == 50
        assert result.column_count == 6

    def test_works_with_numeric_only(self, numeric_only_df: pd.DataFrame):
        result = run_baseline_eda(numeric_only_df)
        assert isinstance(result, BaselineEDAResult)
        assert len(result.univariate) > 0

    def test_works_with_categorical_only(self, categorical_only_df: pd.DataFrame):
        result = run_baseline_eda(categorical_only_df)
        assert isinstance(result, BaselineEDAResult)
        assert len(result.univariate) > 0

