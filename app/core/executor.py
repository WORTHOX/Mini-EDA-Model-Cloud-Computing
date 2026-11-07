"""Whitelisted Pandas/DAX Execution Engine.

This is the CRITICAL security-sensitive module. It executes Structured JSON
Query Plans against a Pandas DataFrame using ONLY whitelisted operations.

Security invariants:
  - NO ``eval()``, NO ``exec()``, NO arbitrary code execution.
  - Every column name is validated against the DataFrame before use.
  - Every filter value is sanitised (type-checked, length-bounded).
  - Only the operators and aggregation functions listed in the whitelist
    are permitted; anything else raises ``ValueError``.

DAX ↔ Pandas mapping (from ``DAX Expressions.pdf``):
  SUM            → df[col].sum()
  AVERAGE        → df[col].mean()
  COUNTROWS      → len(df)
  DISTINCTCOUNT  → df[col].nunique()
  MIN            → df[col].min()
  MAX            → df[col].max()
  STDEV.P        → df[col].std(ddof=0)
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.schemas.json_plan_schema import (
    AggregationSpec,
    FilterCondition,
    QueryPlan,
    SortSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & whitelists
# ---------------------------------------------------------------------------

# Maximum string length accepted for a filter value (anti-injection).
_MAX_FILTER_VALUE_LEN: int = 1024

# Whitelisted comparison operators.
_OPERATOR_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

# Type alias for aggregation functions: each receives (df, column_name).
AggFn = Callable[[pd.DataFrame, str], Any]


def _agg_sum(df: pd.DataFrame, col: str) -> float:
    """DAX SUM → Pandas sum."""
    return float(df[col].sum())


def _agg_mean(df: pd.DataFrame, col: str) -> float:
    """DAX AVERAGE → Pandas mean."""
    return float(df[col].mean())


def _agg_count(df: pd.DataFrame, _col: str) -> int:
    """DAX COUNTROWS → len(df)."""
    return len(df)


def _agg_distinct_count(df: pd.DataFrame, col: str) -> int:
    """DAX DISTINCTCOUNT → Pandas nunique."""
    return int(df[col].nunique())


def _agg_min(df: pd.DataFrame, col: str) -> float:
    """DAX MIN → Pandas min."""
    return float(df[col].min())


def _agg_max(df: pd.DataFrame, col: str) -> float:
    """DAX MAX → Pandas max."""
    return float(df[col].max())


def _agg_std(df: pd.DataFrame, col: str) -> float:
    """DAX STDEV.P → Pandas std(ddof=0) (population standard deviation)."""
    return float(df[col].std(ddof=0))


_AGGREGATION_MAP: Dict[str, AggFn] = {
    "sum": _agg_sum,
    "mean": _agg_mean,
    "count": _agg_count,
    "distinct_count": _agg_distinct_count,
    "min": _agg_min,
    "max": _agg_max,
    "std": _agg_std,
}

# Time-intelligence keywords the plan may specify.
_TIME_INTELLIGENCE_FUNCTIONS = frozenset({"TOTALYTD", "PREVIOUSMONTH"})

# Geographic-related column name fragments used to locate date columns
# when no explicit date column is supplied.
_DATE_KEYWORDS = frozenset(
    {"date", "datetime", "timestamp", "time", "day", "month", "year", "period"}
)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """Structured result returned by the execution engine."""

    result_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tabular result expressed as a list of row-dicts.",
    )
    scalar_result: Optional[float] = Field(
        default=None,
        description="Single scalar value when the query resolves to one number.",
    )
    columns_used: List[str] = Field(
        default_factory=list,
        description="All DataFrame columns referenced during execution.",
    )
    row_count: int = Field(
        default=0,
        description="Number of rows in the result set.",
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (filters applied, agg function, etc.).",
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_column_exists(df: pd.DataFrame, col: str) -> None:
    """Raise ``ValueError`` if *col* is not present in the DataFrame."""
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' does not exist in the DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def _validate_filter_value(value: Any) -> None:
    """Reject dangerously long or exotic filter values (anti-injection)."""
    if isinstance(value, str) and len(value) > _MAX_FILTER_VALUE_LEN:
        raise ValueError(
            f"Filter value exceeds maximum allowed length "
            f"({len(value)} > {_MAX_FILTER_VALUE_LEN})."
        )
    # Only allow primitive scalar types.
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(
            f"Filter value type '{type(value).__name__}' is not permitted."
        )


def _validate_operator(op: str) -> None:
    """Raise ``ValueError`` if *op* is not in the whitelist."""
    if op not in _OPERATOR_MAP:
        raise ValueError(
            f"Operator '{op}' is not whitelisted. "
            f"Allowed: {sorted(_OPERATOR_MAP.keys())}"
        )


def _validate_aggregation(fn: str) -> None:
    """Raise ``ValueError`` if *fn* is not in the whitelist."""
    if fn not in _AGGREGATION_MAP:
        raise ValueError(
            f"Aggregation function '{fn}' is not whitelisted. "
            f"Allowed: {sorted(_AGGREGATION_MAP.keys())}"
        )


# ---------------------------------------------------------------------------
# Date-column auto-detection
# ---------------------------------------------------------------------------


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically locate the best date/datetime column.

    Strategy:
      1. Prefer columns whose dtype is already ``datetime64``.
      2. Fall back to columns whose name contains a date-related keyword
         and that can be successfully parsed by ``pd.to_datetime``.
    """
    # 1. Already-typed datetime columns.
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    if datetime_cols:
        return datetime_cols[0]

    # 2. Name-based heuristic + parseable check.
    for col in df.columns:
        if any(kw in col.lower() for kw in _DATE_KEYWORDS):
            try:
                pd.to_datetime(df[col].dropna().head(20), errors="raise")
                return col
            except (ValueError, TypeError, OverflowError):
                continue

    return None


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------


def _apply_filters(
    df: pd.DataFrame,
    filters: Sequence[FilterCondition],
) -> pd.DataFrame:
    """Apply a sequence of whitelisted filter conditions to *df*.

    Each filter is validated before application:
      - Column must exist.
      - Operator must be whitelisted.
      - Value must pass the injection check.
    """
    for filt in filters:
        _validate_column_exists(df, filt.col)
        _validate_operator(filt.op)
        _validate_filter_value(filt.value)

        op_fn = _OPERATOR_MAP[filt.op]
        mask = op_fn(df[filt.col], filt.value)
        df = df.loc[mask].copy()

    return df


# ---------------------------------------------------------------------------
# Time-intelligence helpers
# ---------------------------------------------------------------------------


def _apply_time_intelligence(
    df: pd.DataFrame,
    function_name: str,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply DAX-style time-intelligence filtering.

    Supported functions:
      * ``TOTALYTD``  – keep rows from the start of the year of the latest
        date through that latest date.
      * ``PREVIOUSMONTH`` – keep rows belonging to the calendar month
        immediately before the latest date's month.

    If *date_col* is ``None`` the engine will attempt auto-detection.
    """
    if function_name not in _TIME_INTELLIGENCE_FUNCTIONS:
        raise ValueError(
            f"Time-intelligence function '{function_name}' is not supported. "
            f"Allowed: {sorted(_TIME_INTELLIGENCE_FUNCTIONS)}"
        )

    # Resolve date column.
    resolved_col = date_col or _find_date_column(df)
    if resolved_col is None:
        raise ValueError(
            "Time-intelligence requested but no date column could be found "
            "in the DataFrame."
        )
    _validate_column_exists(df, resolved_col)

    # Ensure column is datetime-typed.
    if not pd.api.types.is_datetime64_any_dtype(df[resolved_col]):
        df = df.copy()
        df[resolved_col] = pd.to_datetime(df[resolved_col], errors="coerce")

    # Drop rows where the date could not be parsed.
    df = df.dropna(subset=[resolved_col]).copy()

    if df.empty:
        return df

    latest_date: pd.Timestamp = df[resolved_col].max()

    if function_name == "TOTALYTD":
        year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
        df = df.loc[
            (df[resolved_col] >= year_start) & (df[resolved_col] <= latest_date)
        ].copy()

    elif function_name == "PREVIOUSMONTH":
        # Move back one calendar month from the latest date's month.
        if latest_date.month == 1:
            prev_year = latest_date.year - 1
            prev_month = 12
        else:
            prev_year = latest_date.year
            prev_month = latest_date.month - 1

        prev_start = pd.Timestamp(year=prev_year, month=prev_month, day=1)
        # End of that month: start of the next month minus 1 day.
        if prev_month == 12:
            prev_end = pd.Timestamp(year=prev_year + 1, month=1, day=1) - pd.Timedelta(
                days=1
            )
        else:
            prev_end = pd.Timestamp(
                year=prev_year, month=prev_month + 1, day=1
            ) - pd.Timedelta(days=1)

        df = df.loc[
            (df[resolved_col] >= prev_start) & (df[resolved_col] <= prev_end)
        ].copy()

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _apply_aggregation(
    df: pd.DataFrame,
    agg: AggregationSpec,
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame | float | int:
    """Apply a whitelisted aggregation, optionally grouped.

    Returns either a scalar (ungrouped single-column agg) or a DataFrame
    (grouped aggregation).
    """
    _validate_aggregation(agg.fn)
    _validate_column_exists(df, agg.col)

    agg_fn = _AGGREGATION_MAP[agg.fn]

    if group_by:
        for g_col in group_by:
            _validate_column_exists(df, g_col)

        # Avoid column name collision if the aggregated column is in group_by
        agg_result_col = f"{agg.col}_{agg.fn}" if agg.col in group_by else agg.col

        if agg.fn == "count":
            count_col = "Count" if agg.col in group_by else agg.col
            grouped = df.groupby(group_by, sort=False).size().reset_index(name=count_col)
        elif agg.fn == "distinct_count":
            distinct_col = f"{agg.col}_distinct" if agg.col in group_by else agg.col
            grouped = (
                df.groupby(group_by, sort=False)[agg.col]
                .nunique()
                .reset_index(name=distinct_col)
            )
        elif agg.fn == "std":
            std_col = f"{agg.col}_std" if agg.col in group_by else agg.col
            grouped = (
                df.groupby(group_by, sort=False)[agg.col]
                .std(ddof=0)
                .reset_index(name=std_col)
            )
        else:
            pandas_agg_name = {
                "sum": "sum",
                "mean": "mean",
                "min": "min",
                "max": "max",
            }[agg.fn]
            grouped = (
                df.groupby(group_by, sort=False)[agg.col]
                .agg(pandas_agg_name)
                .reset_index(name=agg_result_col)
            )
        return grouped
    else:
        return agg_fn(df, agg.col)


# ---------------------------------------------------------------------------
# Sort & limit
# ---------------------------------------------------------------------------


def _apply_sort(df: pd.DataFrame, sort: SortSpec) -> pd.DataFrame:
    """Sort the DataFrame by the specified column and direction."""
    _validate_column_exists(df, sort.by)
    return df.sort_values(by=sort.by, ascending=not sort.desc).reset_index(drop=True)


def _apply_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Return at most *limit* rows."""
    if limit < 0:
        raise ValueError("Limit must be a non-negative integer.")
    return df.head(limit)


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def execute_query_plan(
    df: pd.DataFrame,
    plan: QueryPlan,
) -> ExecutionResult:
    """Execute a structured JSON query plan against a DataFrame.

    Pipeline order:
      1. Validate all referenced columns exist.
      2. Apply explicit ``filter`` conditions.
      3. Apply ``time_intelligence`` window (TOTALYTD / PREVIOUSMONTH).
      4. Apply ``group_by`` + ``agg`` (aggregation).
      5. Apply ``sort``.
      6. Apply ``limit``.
      7. Package the result into :class:`ExecutionResult`.

    Parameters
    ----------
    df:
        The in-memory Pandas DataFrame to query.
    plan:
        A validated :class:`QueryPlan` instance (Pydantic model).

    Returns
    -------
    ExecutionResult
        Structured result with ``result_data``, ``scalar_result``,
        ``columns_used``, ``row_count``, and ``execution_metadata``.

    Raises
    ------
    ValueError
        If any column, operator, aggregation function, or filter value
        fails validation.
    """
    columns_used: List[str] = []
    metadata: Dict[str, Any] = {
        "operation": plan.operation,
        "filters_applied": [],
        "time_intelligence": None,
        "group_by": None,
        "aggregation": None,
        "sort": None,
        "limit": None,
    }

    working_df = df.copy()

    # ------------------------------------------------------------------
    # 1. Pre-validate all columns referenced in the plan
    # ------------------------------------------------------------------
    if plan.filter:
        for f in plan.filter:
            _validate_column_exists(df, f.col)
            columns_used.append(f.col)

    if plan.agg:
        _validate_column_exists(df, plan.agg.col)
        columns_used.append(plan.agg.col)

    if plan.group_by:
        for g in plan.group_by:
            _validate_column_exists(df, g)
            columns_used.append(g)

    if plan.sort:
        _validate_column_exists(df, plan.sort.by)
        columns_used.append(plan.sort.by)

    # De-duplicate while preserving order.
    columns_used = list(dict.fromkeys(columns_used))

    # ------------------------------------------------------------------
    # 2. Apply filters
    # ------------------------------------------------------------------
    if plan.filter:
        working_df = _apply_filters(working_df, plan.filter)
        metadata["filters_applied"] = [
            {"col": f.col, "op": f.op, "value": f.value} for f in plan.filter
        ]

    # ------------------------------------------------------------------
    # 3. Apply time intelligence
    # ------------------------------------------------------------------
    if plan.time_intelligence:
        ti = plan.time_intelligence.upper()
        working_df = _apply_time_intelligence(working_df, ti)
        metadata["time_intelligence"] = ti

    # ------------------------------------------------------------------
    # 4. Aggregation (with optional group_by)
    # ------------------------------------------------------------------
    scalar_result: Optional[float] = None

    if plan.agg:
        agg_result = _apply_aggregation(
            working_df, plan.agg, group_by=plan.group_by
        )
        metadata["aggregation"] = {"col": plan.agg.col, "fn": plan.agg.fn}

        if plan.group_by:
            metadata["group_by"] = plan.group_by
            # agg_result is a DataFrame.
            working_df = agg_result  # type: ignore[assignment]
        else:
            # Scalar result.
            scalar_result = float(agg_result) if agg_result is not None else None
            # Wrap the scalar in a one-row DataFrame for uniform downstream handling.
            working_df = pd.DataFrame(
                [{plan.agg.col: scalar_result}]
            )
    elif plan.group_by:
        # group_by without agg: just unique grouping (unusual but supported).
        for g in plan.group_by:
            _validate_column_exists(working_df, g)
        working_df = working_df[plan.group_by].drop_duplicates().reset_index(drop=True)
        metadata["group_by"] = plan.group_by

    # ------------------------------------------------------------------
    # 5. Sort
    # ------------------------------------------------------------------
    if plan.sort and isinstance(working_df, pd.DataFrame) and not working_df.empty:
        # Sort column must exist in the (possibly aggregated) result.
        if plan.sort.by in working_df.columns:
            working_df = _apply_sort(working_df, plan.sort)
            metadata["sort"] = {"by": plan.sort.by, "desc": plan.sort.desc}
        else:
            logger.warning(
                "Sort column '%s' not present in aggregated result; skipping sort.",
                plan.sort.by,
            )

    # ------------------------------------------------------------------
    # 6. Limit
    # ------------------------------------------------------------------
    if plan.limit is not None and isinstance(working_df, pd.DataFrame):
        working_df = _apply_limit(working_df, plan.limit)
        metadata["limit"] = plan.limit

    # ------------------------------------------------------------------
    # 7. Build result
    # ------------------------------------------------------------------
    if isinstance(working_df, pd.DataFrame):
        # Replace NaN/NaT with None for clean JSON serialisation.
        result_df = working_df.where(working_df.notna(), other=None)
        result_data = result_df.to_dict(orient="records")
        row_count = len(result_df)
    else:
        result_data = [{"value": working_df}]
        row_count = 1

    return ExecutionResult(
        result_data=result_data,
        scalar_result=scalar_result,
        columns_used=columns_used,
        row_count=row_count,
        execution_metadata=metadata,
    )
