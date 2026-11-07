"""Company-Standard Visual Selection Engine.

Maps a combination of analytical intent + data shape to the single best
company-approved ``ChartType``.  The mapping table is a codified version of
the Power BI / Tableau visual selection best-practice matrix:

    Category vs Metric Comparison       → BAR
    Metric Trend Over Time              → LINE or COLUMN
    Cumulative Metric Growth            → AREA
    Single Important Figure             → CARD
    Multiple Metrics Comparison         → MULTI_ROW_CARD
    Proportion / Share Analysis         → DONUT or PIE
    Two Numeric Variables (Correlation) → SCATTER
    Three/Four Numeric Variables        → BUBBLE
    Single Variable Distribution        → HISTOGRAM
    Multi-Level Hierarchical Breakdown  → MATRIX or TREEMAP
    Sequential Incremental Impact       → WATERFALL
    Target Goal Tracking                → GAUGE
    Spatial / Geographic Plotting       → MAP
"""

from __future__ import annotations

from typing import List

from app.core.field_mapper import ColumnMeta, DtypeClass
from app.schemas.json_plan_schema import AggFn, OperationType, QueryPlan
from app.schemas.visual_payload import ChartType


# ---------------------------------------------------------------------------
# Canonical intent labels
# ---------------------------------------------------------------------------

INTENT_CATEGORY_COMPARISON = "category_comparison"
INTENT_TIME_TREND = "time_trend"
INTENT_CUMULATIVE_GROWTH = "cumulative_growth"
INTENT_SINGLE_VALUE = "single_value"
INTENT_MULTI_METRIC = "multi_metric"
INTENT_PROPORTION = "proportion"
INTENT_CORRELATION = "correlation"
INTENT_MULTI_NUMERIC = "multi_numeric"
INTENT_DISTRIBUTION = "distribution"
INTENT_HIERARCHICAL = "hierarchical"
INTENT_WATERFALL = "waterfall"
INTENT_GOAL_TRACKING = "goal_tracking"
INTENT_GEOGRAPHIC = "geographic"
INTENT_RAW_TABLE = "raw_table"


# ---------------------------------------------------------------------------
# Intent → ChartType mapping table
# ---------------------------------------------------------------------------

_INTENT_MAP: dict[str, ChartType] = {
    INTENT_CATEGORY_COMPARISON: ChartType.BAR,
    INTENT_TIME_TREND: ChartType.LINE,
    INTENT_CUMULATIVE_GROWTH: ChartType.AREA,
    INTENT_SINGLE_VALUE: ChartType.CARD,
    INTENT_MULTI_METRIC: ChartType.MULTI_ROW_CARD,
    INTENT_PROPORTION: ChartType.DONUT,
    INTENT_CORRELATION: ChartType.SCATTER,
    INTENT_MULTI_NUMERIC: ChartType.BUBBLE,
    INTENT_DISTRIBUTION: ChartType.HISTOGRAM,
    INTENT_HIERARCHICAL: ChartType.TREEMAP,
    INTENT_WATERFALL: ChartType.WATERFALL,
    INTENT_GOAL_TRACKING: ChartType.GAUGE,
    INTENT_GEOGRAPHIC: ChartType.MAP,
    INTENT_RAW_TABLE: ChartType.MATRIX,
}


# ---------------------------------------------------------------------------
# Data-shape helpers
# ---------------------------------------------------------------------------

def _count_by_dtype(
    columns: List[ColumnMeta],
) -> tuple[int, int, bool, bool]:
    """Return (num_numeric, num_categorical, has_date, has_geo)."""
    num_numeric = sum(1 for c in columns if c.dtype_class == DtypeClass.NUMERIC)
    num_categorical = sum(1 for c in columns if c.dtype_class == DtypeClass.TEXT)
    has_date = any(c.dtype_class == DtypeClass.DATE for c in columns)
    has_geo = any(c.dtype_class == DtypeClass.GEO for c in columns)
    return num_numeric, num_categorical, has_date, has_geo


# ---------------------------------------------------------------------------
# Public API — intent inference
# ---------------------------------------------------------------------------

def infer_intent(
    query_plan: QueryPlan,
    columns: List[ColumnMeta],
) -> str:
    """Determine the analytical intent from a JSON query plan and columns.

    The function examines the plan's operation type, aggregation function,
    group-by columns, and the data-type profile of involved columns to
    produce a canonical intent label.

    Args:
        query_plan: The validated structured query plan.
        columns: Column metadata descriptors for the data involved.

    Returns:
        A canonical intent string (one of the ``INTENT_*`` constants).
    """
    num_numeric, num_categorical, has_date, has_geo = _count_by_dtype(columns)

    # --- Geographic: only if the plan actually groups by or filters on a geo column ---
    if has_geo:
        geo_col_names = {c.name for c in columns if c.dtype_class == DtypeClass.GEO}
        plan_cols = set(query_plan.group_by or [])
        if query_plan.agg:
            plan_cols.add(query_plan.agg.col)
        for f in query_plan.filter:
            plan_cols.add(f.col)
        if plan_cols & geo_col_names:
            return INTENT_GEOGRAPHIC

    # --- Raw operation — just show the table ---
    if query_plan.operation == OperationType.RAW:
        return INTENT_RAW_TABLE

    # --- Filter-only (no aggregation) ---
    if query_plan.operation == OperationType.FILTER:
        if num_numeric == 0:
            return INTENT_RAW_TABLE
        if has_date:
            return INTENT_TIME_TREND
        return INTENT_CATEGORY_COMPARISON

    # --- Sort-only ---
    if query_plan.operation == OperationType.SORT:
        if has_date:
            return INTENT_TIME_TREND
        return INTENT_CATEGORY_COMPARISON

    # --- Aggregation — the richest branch ---
    if query_plan.operation == OperationType.AGGREGATION:
        agg = query_plan.agg  # guaranteed non-None by model validator
        assert agg is not None  # for type checker

        # Single scalar result (no group_by)
        if not query_plan.group_by:
            if num_numeric > 1:
                return INTENT_MULTI_METRIC
            return INTENT_SINGLE_VALUE

        # Count / distinct_count on a single group → distribution / proportion
        if agg.fn in (AggFn.COUNT, AggFn.DISTINCT_COUNT):
            if num_categorical >= 1 and num_numeric <= 1:
                return INTENT_PROPORTION

        # Time-intelligence modifier present
        if query_plan.time_intelligence is not None:
            return INTENT_CUMULATIVE_GROWTH

        # Date in group_by → time trend
        group_by_set = set(query_plan.group_by)
        date_cols = {c.name for c in columns if c.dtype_class == DtypeClass.DATE}
        if group_by_set & date_cols:
            return INTENT_TIME_TREND

        # Multiple group_by levels → hierarchical
        if len(query_plan.group_by) >= 2:
            return INTENT_HIERARCHICAL

        # Standard deviation → distribution
        if agg.fn == AggFn.STD:
            return INTENT_DISTRIBUTION

        # Multiple numerics → correlation or multi-numeric
        if num_numeric >= 3:
            return INTENT_MULTI_NUMERIC
        if num_numeric == 2:
            return INTENT_CORRELATION

        # Default aggregation with category → category comparison
        if num_categorical >= 1:
            return INTENT_CATEGORY_COMPARISON

    # Catch-all
    return INTENT_RAW_TABLE


# ---------------------------------------------------------------------------
# Public API — visual selection
# ---------------------------------------------------------------------------

def select_visual(
    intent: str,
    columns: List[ColumnMeta],
    has_date: bool | None = None,
    has_geo: bool | None = None,
    num_numeric: int | None = None,
    num_categorical: int | None = None,
) -> ChartType:
    """Select the best company-approved chart type for the given intent + data shape.

    The function first checks the canonical intent → chart-type mapping table,
    then applies data-shape refinements (e.g. using COLUMN instead of LINE
    when there are few time points).

    Args:
        intent: Canonical intent label (from ``infer_intent``).
        columns: Column metadata descriptors.
        has_date: Whether date columns are present (auto-detected if ``None``).
        has_geo: Whether geographic columns are present (auto-detected if ``None``).
        num_numeric: Count of numeric columns (auto-detected if ``None``).
        num_categorical: Count of categorical columns (auto-detected if ``None``).

    Returns:
        The best matching ``ChartType``.
    """
    # Auto-detect shape if not provided
    if any(v is None for v in (has_date, has_geo, num_numeric, num_categorical)):
        _nn, _nc, _hd, _hg = _count_by_dtype(columns)
        if num_numeric is None:
            num_numeric = _nn
        if num_categorical is None:
            num_categorical = _nc
        if has_date is None:
            has_date = _hd
        if has_geo is None:
            has_geo = _hg

    # Base chart from intent table
    chart = _INTENT_MAP.get(intent, ChartType.MATRIX)

    # --- Data-shape refinements ---

    # Time trend: prefer COLUMN when only 1 categorical and no date
    if intent == INTENT_TIME_TREND and not has_date and num_categorical >= 1:
        chart = ChartType.COLUMN

    # Proportion: use PIE when categories ≤ 5, DONUT otherwise
    if intent == INTENT_PROPORTION and num_categorical >= 1:
        chart = ChartType.PIE if num_categorical <= 5 else ChartType.DONUT

    # Multi-metric with categories → CLUSTERED_COLUMN
    if intent == INTENT_MULTI_METRIC and num_categorical >= 1:
        chart = ChartType.CLUSTERED_COLUMN

    # Category comparison with multiple measures → CLUSTERED_BAR
    if intent == INTENT_CATEGORY_COMPARISON and num_numeric >= 2:
        chart = ChartType.CLUSTERED_BAR

    # Hierarchical: MATRIX for 3+ levels, TREEMAP for 2
    if intent == INTENT_HIERARCHICAL and num_categorical >= 3:
        chart = ChartType.MATRIX

    return chart
