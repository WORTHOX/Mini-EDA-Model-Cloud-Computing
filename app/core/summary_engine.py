"""Grounded Summary Engine.

Generates concise, factual 1-2 line natural-language summaries of computed
analytical results.  Every number referenced in the summary is sourced
directly from the computation output — **no hallucinated or inferred
statistics**.

Design principles:
    • Strictly grounded: only reference numbers present in the result data.
    • Human-readable formatting: $1.42M, 14.2%, 1,234 units.
    • Context-aware: adapts phrasing to chart type and aggregation function.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.schemas.json_plan_schema import AggFn, OperationType, QueryPlan
from app.schemas.visual_payload import ChartType


# ---------------------------------------------------------------------------
# Number formatting utilities
# ---------------------------------------------------------------------------

def _format_number(value: Any, is_currency: bool = False) -> str:
    """Format a numeric value for human-readable display.

    Examples:
        1234567   → "1.23M"
        0.1423    → "14.2%"   (if detected as ratio)
        1234      → "1,234"

    Args:
        value: The numeric value to format.
        is_currency: If ``True``, prefix with ``$``.

    Returns:
        A formatted string representation.
    """
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"

    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    prefix = "$" if is_currency else ""

    abs_num = abs(num)

    if abs_num >= 1_000_000_000:
        return f"{prefix}{num / 1_000_000_000:,.2f}B"
    if abs_num >= 1_000_000:
        return f"{prefix}{num / 1_000_000:,.2f}M"
    if abs_num >= 1_000:
        return f"{prefix}{num:,.0f}"
    if abs_num == 0:
        return f"{prefix}0"
    if abs_num < 1:
        # Could be a ratio / percentage
        return f"{num * 100:.1f}%"

    # Small numbers with decimals
    if num == int(num):
        return f"{prefix}{int(num)}"
    return f"{prefix}{num:,.2f}"


def _detect_currency(col_name: str) -> bool:
    """Heuristic: detect if a column name suggests currency values."""
    currency_keywords = {
        "revenue", "sales", "price", "cost", "profit", "amount",
        "income", "expense", "budget", "spend", "fee", "salary",
        "wage", "payment", "total", "value", "margin", "arpu",
    }
    return any(kw in col_name.lower() for kw in currency_keywords)


# ---------------------------------------------------------------------------
# Summary template helpers
# ---------------------------------------------------------------------------

def _agg_fn_verb(fn: AggFn) -> str:
    """Map aggregation function to a human-readable verb/noun."""
    return {
        AggFn.SUM: "total",
        AggFn.MEAN: "average",
        AggFn.COUNT: "count",
        AggFn.DISTINCT_COUNT: "distinct count",
        AggFn.MIN: "minimum",
        AggFn.MAX: "maximum",
        AggFn.STD: "standard deviation",
    }.get(fn, str(fn.value))


def _chart_type_label(chart_type: ChartType) -> str:
    """Friendly label for a chart type (used in fallback summaries)."""
    return chart_type.value.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Core summarisation logic
# ---------------------------------------------------------------------------

def _summarise_scalar(
    result_data: Any,
    query_plan: QueryPlan,
    columns: List[str],
) -> str:
    """Generate summary for a single-scalar result (e.g. overall total)."""
    agg = query_plan.agg
    if agg is None:
        return f"The result is {_format_number(result_data)}."

    is_curr = _detect_currency(agg.col)
    verb = _agg_fn_verb(agg.fn)
    formatted = _format_number(result_data, is_currency=is_curr)
    return f"The {verb} of {agg.col} is {formatted}."


def _summarise_grouped(
    df: pd.DataFrame,
    query_plan: QueryPlan,
    columns: List[str],
) -> str:
    """Generate summary for a grouped aggregation result."""
    agg = query_plan.agg
    if agg is None or df.empty:
        return f"The query returned {len(df)} rows."

    is_curr = _detect_currency(agg.col)
    verb = _agg_fn_verb(agg.fn)
    measure_col = agg.col

    # Find the aggregated values column in the DataFrame
    # It might be named exactly as agg.col or with a suffix
    agg_col_name: str | None = None
    for c in df.columns:
        if c == measure_col or measure_col in c:
            if df[c].dtype.kind in ("i", "f"):
                agg_col_name = c
                break

    if agg_col_name is None:
        # Fallback: pick the first numeric column
        for c in df.columns:
            if df[c].dtype.kind in ("i", "f"):
                agg_col_name = c
                break

    if agg_col_name is None:
        return f"The query returned {len(df)} grouped results."

    group_col = query_plan.group_by[0] if query_plan.group_by else None

    # Top value
    top_idx = df[agg_col_name].idxmax()
    top_val = df.at[top_idx, agg_col_name]
    top_label = df.at[top_idx, group_col] if group_col and group_col in df.columns else None

    formatted_top = _format_number(top_val, is_currency=is_curr)

    parts: list[str] = []

    if top_label is not None:
        parts.append(
            f"{top_label} has the highest {verb} of {measure_col} at {formatted_top}."
        )
    else:
        parts.append(
            f"The highest {verb} of {measure_col} is {formatted_top}."
        )

    # If there's a bottom value and more than 1 row, include range context
    if len(df) > 1:
        bottom_idx = df[agg_col_name].idxmin()
        bottom_val = df.at[bottom_idx, agg_col_name]
        bottom_label = (
            df.at[bottom_idx, group_col]
            if group_col and group_col in df.columns
            else None
        )
        formatted_bottom = _format_number(bottom_val, is_currency=is_curr)
        if bottom_label is not None:
            parts.append(
                f"The lowest is {bottom_label} at {formatted_bottom} across {len(df)} groups."
            )
        else:
            parts.append(
                f"Values range down to {formatted_bottom} across {len(df)} groups."
            )

    return " ".join(parts)


def _summarise_filtered(
    df: pd.DataFrame,
    query_plan: QueryPlan,
    columns: List[str],
) -> str:
    """Generate summary for a filter / sort result."""
    filter_parts: list[str] = []
    for f in query_plan.filter:
        filter_parts.append(f"{f.col} {f.op.value} {f.value}")

    filter_desc = " and ".join(filter_parts) if filter_parts else "the applied filters"
    return f"Showing {len(df)} rows where {filter_desc}."


def _summarise_time_series(
    df: pd.DataFrame,
    query_plan: QueryPlan,
    columns: List[str],
) -> str:
    """Generate summary for time-series / trend data."""
    agg = query_plan.agg
    if agg is None or df.empty:
        return f"Time series shows {len(df)} data points."

    is_curr = _detect_currency(agg.col)
    verb = _agg_fn_verb(agg.fn)

    # Find the numeric column
    agg_col_name: str | None = None
    for c in df.columns:
        if c == agg.col or agg.col in c:
            if df[c].dtype.kind in ("i", "f"):
                agg_col_name = c
                break

    if agg_col_name is None:
        return f"Time trend shows {len(df)} data points for {agg.col}."

    first_val = df[agg_col_name].iloc[0]
    last_val = df[agg_col_name].iloc[-1]

    formatted_first = _format_number(first_val, is_currency=is_curr)
    formatted_last = _format_number(last_val, is_currency=is_curr)

    if first_val != 0:
        change_pct = ((last_val - first_val) / abs(first_val)) * 100
        direction = "increased" if change_pct > 0 else "decreased"
        return (
            f"The {verb} of {agg.col} {direction} from {formatted_first} "
            f"to {formatted_last} ({abs(change_pct):.1f}% change) over {len(df)} periods."
        )

    return (
        f"The {verb} of {agg.col} moved from {formatted_first} "
        f"to {formatted_last} over {len(df)} periods."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(
    result_data: Union[pd.DataFrame, Any],
    query_plan: QueryPlan,
    chart_type: ChartType,
    columns: List[str],
) -> str:
    """Generate a grounded 1-2 line natural-language summary.

    The summary strictly references only computed numbers from
    ``result_data``.  No ungrounded statements are produced.

    Args:
        result_data: The computation result — either a ``pd.DataFrame`` of
            result rows, or a scalar value for single-figure results.
        query_plan: The validated query plan that produced the result.
        chart_type: The selected chart type (used for context-aware phrasing).
        columns: Column names present in the result.

    Returns:
        A concise, factual summary string.
    """
    # --- Scalar result ---
    if not isinstance(result_data, pd.DataFrame):
        return _summarise_scalar(result_data, query_plan, columns)

    df = result_data

    if df.empty:
        return "The query returned no results."

    # --- Single-row result → treat as scalar ---
    if len(df) == 1 and len(df.columns) == 1:
        return _summarise_scalar(df.iloc[0, 0], query_plan, columns)

    # --- Time-series charts ---
    if chart_type in (ChartType.LINE, ChartType.AREA):
        return _summarise_time_series(df, query_plan, columns)

    # --- Aggregation with groups ---
    if (
        query_plan.operation == OperationType.AGGREGATION
        and query_plan.group_by
    ):
        return _summarise_grouped(df, query_plan, columns)

    # --- Filter / Sort operations ---
    if query_plan.operation in (OperationType.FILTER, OperationType.SORT):
        return _summarise_filtered(df, query_plan, columns)

    # --- CARD / GAUGE: single important figure ---
    if chart_type in (ChartType.CARD, ChartType.GAUGE):
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in ("i", "f")]
        if numeric_cols:
            val = df[numeric_cols[0]].iloc[0]
            is_curr = _detect_currency(numeric_cols[0])
            formatted = _format_number(val, is_currency=is_curr)
            return f"{numeric_cols[0]}: {formatted}."

    # --- Default fallback ---
    row_count = len(df)
    col_count = len(df.columns)
    return f"Displaying {row_count} rows across {col_count} columns."
