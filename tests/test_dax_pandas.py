"""Tests for DAX-to-Pandas Translation (Whitelisted Execution Engine).

Validates that every DAX formula maps correctly to its Pandas equivalent
and that security invariants (no eval/exec, whitelisted-only operations)
are maintained.
"""

import pandas as pd
import pytest

from app.core.executor import ExecutionResult, execute_query_plan
from app.schemas.json_plan_schema import (
    AggregationSpec,
    FilterCondition,
    FilterOp,
    OperationType,
    QueryPlan,
    SortSpec,
    TimeIntelligence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """A sales-like DataFrame for testing DAX translations."""
    return pd.DataFrame(
        {
            "Region": ["North", "South", "East", "West"] * 25,
            "Product": [f"P{i % 5}" for i in range(100)],
            "Revenue": [float(100 + i * 5) for i in range(100)],
            "Quantity": [int(1 + i % 10) for i in range(100)],
            "Status": ["Completed", "Pending", "Completed", "Cancelled"] * 25,
            "OrderDate": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
    )


# ---------------------------------------------------------------------------
# DAX SUM → Pandas sum
# ---------------------------------------------------------------------------


class TestDAXSum:
    def test_sum_no_filter(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="sum"),
        )
        result = execute_query_plan(sales_df, plan)
        expected = sales_df["Revenue"].sum()
        assert result.scalar_result == pytest.approx(expected)

    def test_sum_with_filter(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            filter=[FilterCondition(col="Status", op=FilterOp.EQ, value="Completed")],
            agg=AggregationSpec(col="Revenue", fn="sum"),
        )
        result = execute_query_plan(sales_df, plan)
        expected = sales_df[sales_df["Status"] == "Completed"]["Revenue"].sum()
        assert result.scalar_result == pytest.approx(expected)

    def test_sum_grouped(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            group_by=["Region"],
            agg=AggregationSpec(col="Revenue", fn="sum"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == 4  # 4 regions
        total = sum(r["Revenue"] for r in result.result_data)
        assert total == pytest.approx(sales_df["Revenue"].sum())


# ---------------------------------------------------------------------------
# DAX AVERAGE → Pandas mean
# ---------------------------------------------------------------------------


class TestDAXAverage:
    def test_mean_global(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="mean"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result == pytest.approx(sales_df["Revenue"].mean())

    def test_mean_grouped(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            group_by=["Region"],
            agg=AggregationSpec(col="Revenue", fn="mean"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == 4


# ---------------------------------------------------------------------------
# DAX COUNTROWS → len(df)
# ---------------------------------------------------------------------------


class TestDAXCount:
    def test_count_all(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="count"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result == pytest.approx(100.0)

    def test_count_filtered(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            filter=[FilterCondition(col="Region", op=FilterOp.EQ, value="North")],
            agg=AggregationSpec(col="Revenue", fn="count"),
        )
        result = execute_query_plan(sales_df, plan)
        expected = len(sales_df[sales_df["Region"] == "North"])
        assert result.scalar_result == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# DAX DISTINCTCOUNT → nunique
# ---------------------------------------------------------------------------


class TestDAXDistinctCount:
    def test_distinct_count(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Product", fn="distinct_count"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result == pytest.approx(float(sales_df["Product"].nunique()))


# ---------------------------------------------------------------------------
# DAX MIN / MAX → Pandas min / max
# ---------------------------------------------------------------------------


class TestDAXMinMax:
    def test_min(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="min"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result == pytest.approx(sales_df["Revenue"].min())

    def test_max(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="max"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result == pytest.approx(sales_df["Revenue"].max())


# ---------------------------------------------------------------------------
# DAX STDEV.P → Pandas std(ddof=0)
# ---------------------------------------------------------------------------


class TestDAXStdevP:
    def test_population_std(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="std"),
        )
        result = execute_query_plan(sales_df, plan)
        expected = sales_df["Revenue"].std(ddof=0)
        assert result.scalar_result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Filter operator tests
# ---------------------------------------------------------------------------


class TestFilterOperators:
    @pytest.mark.parametrize("op,expected_count", [
        (FilterOp.EQ, 25),    # "North" appears 25 times
        (FilterOp.NE, 75),    # Not "North" = 75
    ])
    def test_string_filters(self, sales_df, op, expected_count):
        plan = QueryPlan(
            operation=OperationType.FILTER,
            filter=[FilterCondition(col="Region", op=op, value="North")],
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == expected_count

    def test_numeric_gt(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.FILTER,
            filter=[FilterCondition(col="Revenue", op=FilterOp.GT, value=400.0)],
        )
        result = execute_query_plan(sales_df, plan)
        expected = len(sales_df[sales_df["Revenue"] > 400.0])
        assert result.row_count == expected


# ---------------------------------------------------------------------------
# Sort and limit tests
# ---------------------------------------------------------------------------


class TestSortLimit:
    def test_sort_desc_limit(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.SORT,
            sort=SortSpec(by="Revenue", desc=True),
            limit=5,
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == 5
        revenues = [r["Revenue"] for r in result.result_data]
        assert revenues == sorted(revenues, reverse=True)

    def test_sort_asc(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.SORT,
            sort=SortSpec(by="Revenue", desc=False),
            limit=3,
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == 3
        revenues = [r["Revenue"] for r in result.result_data]
        assert revenues == sorted(revenues)


# ---------------------------------------------------------------------------
# Time intelligence tests
# ---------------------------------------------------------------------------


class TestTimeIntelligence:
    def test_totalytd(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="sum"),
            time_intelligence="TOTALYTD",
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result is not None
        assert result.scalar_result > 0

    def test_previousmonth(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="Revenue", fn="sum"),
            time_intelligence="PREVIOUSMONTH",
        )
        result = execute_query_plan(sales_df, plan)
        assert result.scalar_result is not None


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_rejects_unknown_column(self, sales_df: pd.DataFrame):
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            agg=AggregationSpec(col="NonExistentColumn", fn="sum"),
        )
        with pytest.raises(ValueError, match="does not exist"):
            execute_query_plan(sales_df, plan)

    def test_rejects_invalid_agg_function(self):
        with pytest.raises(ValueError):
            AggregationSpec(col="Revenue", fn="eval")

    def test_rejects_invalid_filter_op(self):
        with pytest.raises(ValueError):
            FilterCondition(col="Revenue", op="DROP TABLE", value=0)


# ---------------------------------------------------------------------------
# Collision and Complex Query Tests
# ---------------------------------------------------------------------------


class TestComplexQueries:
    def test_count_grouped_same_column(self, sales_df: pd.DataFrame):
        """Count grouped by the counted column itself (no index collision)."""
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            group_by=["Product"],
            agg=AggregationSpec(col="Product", fn="count"),
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == sales_df["Product"].nunique()
        assert "Count" in result.result_data[0]
        assert "Product" in result.columns_used
        total = sum(r["Count"] for r in result.result_data)
        assert total == len(sales_df)

    def test_top_n_grouped(self, sales_df: pd.DataFrame):
        """Top N items by sum of metric."""
        plan = QueryPlan(
            operation=OperationType.AGGREGATION,
            group_by=["Product"],
            agg=AggregationSpec(col="Revenue", fn="sum"),
            sort=SortSpec(by="Revenue", desc=True),
            limit=3,
        )
        result = execute_query_plan(sales_df, plan)
        assert result.row_count == 3
        revs = [r["Revenue"] for r in result.result_data]
        assert revs == sorted(revs, reverse=True)


