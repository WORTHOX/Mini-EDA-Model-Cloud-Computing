"""Automated Baseline EDA Generator (Page 1).

On CSV upload this module automatically produces three tiers of analysis:

  * **Tier 1 – Univariate**: KPI cards, histograms, bar-frequency charts.
  * **Tier 2 – Bivariate**: time-series lines, clustered bars, scatter plots.
  * **Tier 3 – Multivariate**: bubble charts, pivot/matrix tables, stacked
    bar/area charts.

All numeric computations use the DAX-mapped Pandas functions defined in the
project specification (``DAX Expressions.pdf``).

Visual payloads are emitted as :class:`ChartPayload` objects that the frontend
can render directly via Plotly / Chart.js / any charting library.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.schemas.visual_payload import ChartPayload, ChartType, FieldPlacement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_MAX_TEXT_UNIQUES_BAR: int = 30
_MAX_TEXT_UNIQUES_BIVARIATE: int = 20
_MAX_SCATTER_PAIRS: int = 6
_MAX_BUBBLE_CHARTS: int = 2

_GEO_KEYWORDS: frozenset[str] = frozenset({
    "lat", "latitude", "lon", "lng", "longitude", "city", "state",
    "country", "zip", "zipcode", "zip_code", "postal", "region",
    "location", "address", "county", "province", "district", "geo",
})
_DATE_KEYWORDS: frozenset[str] = frozenset({
    "date", "datetime", "timestamp", "time", "day", "month", "year", "period",
})


# ---------------------------------------------------------------------------
# Column metadata model
# ---------------------------------------------------------------------------


class ColumnMeta(BaseModel):
    """Schema / metadata descriptor for a single DataFrame column."""
    name: str
    dtype_class: Literal["TEXT", "NUMERIC", "DATE", "GEO"]
    sample_values: List[Any] = Field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------


def _is_geo_column(col_name: str) -> bool:
    """Return ``True`` if the column name matches a geographic keyword."""
    lower = col_name.lower()
    return any(kw == lower or kw in lower.replace("_", " ").split() for kw in _GEO_KEYWORDS)


def _try_parse_datetime(series: pd.Series) -> bool:
    """Return ``True`` if the series can be parsed as datetime.

    Uses strict format inference to avoid false positives on plain text
    columns like 'North', 'South' which dateutil can parse ambiguously.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        sample = series.dropna().head(50).astype(str)
        if sample.empty:
            return False
        # Require the column name to contain a date-like keyword,
        # OR the values themselves to contain digits (date-like patterns).
        col_name_lower = series.name.lower() if series.name else ""
        has_date_keyword = any(kw in col_name_lower for kw in _DATE_KEYWORDS)
        has_digit_values = sample.str.contains(r"\d", regex=True).mean() > 0.5

        if not has_date_keyword and not has_digit_values:
            return False

        pd.to_datetime(sample, errors="raise", format="mixed")
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def classify_columns(df: pd.DataFrame) -> List[ColumnMeta]:
    """Classify every column in *df* into TEXT, NUMERIC, DATE, or GEO.

    Classification priority (first match wins):
      1. **NUMERIC** – ``int64``, ``float64``, or other numeric dtypes.
      2. **DATE** – already ``datetime64`` or parseable by ``pd.to_datetime``.
      3. **GEO** – column name matches geographic keywords.
      4. **TEXT** – everything else.
    """
    metas: List[ColumnMeta] = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique())
        raw_samples = series.dropna().head(5).tolist()
        sample_values = [
            v if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else None
            for v in raw_samples
        ]

        if pd.api.types.is_numeric_dtype(series):
            dtype_class: Literal["TEXT", "NUMERIC", "DATE", "GEO"] = "NUMERIC"
        elif _try_parse_datetime(series):
            dtype_class = "DATE"
        elif _is_geo_column(col):
            dtype_class = "GEO"
        else:
            dtype_class = "TEXT"

        metas.append(ColumnMeta(
            name=col,
            dtype_class=dtype_class,
            sample_values=sample_values,
            null_count=null_count,
            unique_count=unique_count,
        ))

    return metas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Optional[float]:
    """Convert *v* to ``float``, returning ``None`` on failure."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (ValueError, TypeError):
        return None


def _fmt(val: Any) -> str:
    """Format a number for summary text."""
    if val is None:
        return "N/A"
    try:
        n = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    if isinstance(val, int) or n == int(n):
        return f"{int(n):,}"
    return f"{n:,.2f}"


# ---------------------------------------------------------------------------
# Tier 1 – Univariate Analysis
# ---------------------------------------------------------------------------


def generate_univariate(
    df: pd.DataFrame,
    columns: List[ColumnMeta],
) -> List[ChartPayload]:
    """Generate Tier-1 univariate chart payloads.

    For each NUMERIC column:
      * A **KPI Card** with sum, mean, min, max, count, std.
      * A **Histogram** of the distribution.

    For each TEXT column (unique count < 30):
      * A **Bar Frequency** chart of value counts.
    """
    payloads: List[ChartPayload] = []

    for meta in columns:
        if meta.dtype_class == "NUMERIC":
            series = df[meta.name].dropna()

            # ── KPI Card ──────────────────────────────────────────
            kpi_stats = {
                "count": len(df),
                "sum": _safe_float(series.sum()),
                "mean": _safe_float(series.mean()),
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "std": _safe_float(series.std(ddof=0)),
                "distinct_count": int(series.nunique()),
            }

            payloads.append(ChartPayload(
                chart_type=ChartType.CARD,
                title=f"KPI Summary – {meta.name}",
                field_placement=FieldPlacement(values=meta.name),
                data=[kpi_stats],
                summary_text=(
                    f"{meta.name}: sum={_fmt(kpi_stats['sum'])}, "
                    f"mean={_fmt(kpi_stats['mean'])}, "
                    f"count={kpi_stats['count']:,}."
                ),
            ))

            # ── Histogram ─────────────────────────────────────────
            if len(series) > 0:
                hist_data = series.tolist()
                payloads.append(ChartPayload(
                    chart_type=ChartType.HISTOGRAM,
                    title=f"Distribution – {meta.name}",
                    field_placement=FieldPlacement(axis_x=meta.name),
                    data=[{meta.name: v} for v in hist_data],
                    summary_text=f"Distribution of {meta.name} across {len(hist_data)} values.",
                ))

        elif meta.dtype_class == "TEXT" and meta.unique_count < _MAX_TEXT_UNIQUES_BAR:
            # ── Bar Frequency ─────────────────────────────────────
            vc = df[meta.name].value_counts().head(_MAX_TEXT_UNIQUES_BAR)
            bar_data = [
                {meta.name: str(label), "Count": int(count)}
                for label, count in zip(vc.index, vc.values)
            ]

            payloads.append(ChartPayload(
                chart_type=ChartType.BAR,
                title=f"Frequency – {meta.name}",
                field_placement=FieldPlacement(axis_y=meta.name, axis_x="Count"),
                data=bar_data,
                summary_text=f"Top value: {vc.index[0]} ({vc.values[0]:,} occurrences)." if len(vc) > 0 else "",
            ))

    return payloads


# ---------------------------------------------------------------------------
# Tier 2 – Bivariate Analysis
# ---------------------------------------------------------------------------


def generate_bivariate(
    df: pd.DataFrame,
    columns: List[ColumnMeta],
) -> List[ChartPayload]:
    """Generate Tier-2 bivariate chart payloads.

    * DATE × NUMERIC → Line Chart (time-series trend).
    * TEXT × NUMERIC (TEXT unique < 20) → Clustered Bar Chart.
    * NUMERIC × NUMERIC (first few pairs) → Scatter Plot with correlation.
    """
    payloads: List[ChartPayload] = []

    date_cols = [c for c in columns if c.dtype_class == "DATE"]
    num_cols = [c for c in columns if c.dtype_class == "NUMERIC"]
    text_cols = [
        c for c in columns
        if c.dtype_class == "TEXT" and c.unique_count < _MAX_TEXT_UNIQUES_BIVARIATE
    ]

    # ── DATE × NUMERIC → Line Chart ────────────────────────────
    for d_meta in date_cols:
        date_series = pd.to_datetime(df[d_meta.name], errors="coerce")
        for n_meta in num_cols:
            temp = pd.DataFrame({"date": date_series, "value": df[n_meta.name]}).dropna()
            if temp.empty:
                continue
            temp = temp.sort_values("date")
            agg = temp.groupby("date")["value"].sum().reset_index()
            line_data = [
                {d_meta.name: str(row["date"]), n_meta.name: float(row["value"])}
                for _, row in agg.iterrows()
            ]

            payloads.append(ChartPayload(
                chart_type=ChartType.LINE,
                title=f"Trend – {n_meta.name} over {d_meta.name}",
                field_placement=FieldPlacement(axis_x=d_meta.name, axis_y=n_meta.name),
                data=line_data,
                summary_text=(
                    f"{n_meta.name} over {d_meta.name}: "
                    f"{len(line_data)} data points."
                ),
            ))

    # ── TEXT × NUMERIC → Clustered Bar Chart ──────────────────
    for t_meta in text_cols:
        for n_meta in num_cols:
            grouped = (
                df.groupby(t_meta.name, sort=False)[n_meta.name]
                .sum()
                .reset_index()
                .sort_values(n_meta.name, ascending=False)
                .head(_MAX_TEXT_UNIQUES_BIVARIATE)
            )
            if grouped.empty:
                continue

            bar_data = [
                {t_meta.name: str(row[t_meta.name]), n_meta.name: float(row[n_meta.name])}
                for _, row in grouped.iterrows()
            ]
            payloads.append(ChartPayload(
                chart_type=ChartType.CLUSTERED_BAR,
                title=f"{n_meta.name} by {t_meta.name}",
                field_placement=FieldPlacement(axis_y=t_meta.name, axis_x=n_meta.name),
                data=bar_data,
                summary_text=(
                    f"Top {t_meta.name}: {grouped.iloc[0][t_meta.name]} "
                    f"({_fmt(grouped.iloc[0][n_meta.name])})."
                ),
            ))

    # ── NUMERIC × NUMERIC → Scatter Plot ──────────────────────
    scatter_count = 0
    for col_a, col_b in combinations(num_cols, 2):
        if scatter_count >= _MAX_SCATTER_PAIRS:
            break
        pair_df = df[[col_a.name, col_b.name]].dropna()
        if len(pair_df) < 3:
            continue
        corr = pair_df[col_a.name].corr(pair_df[col_b.name])
        corr_str = f"{corr:.3f}" if not (math.isnan(corr) or math.isinf(corr)) else "N/A"
        scatter_data = [
            {col_a.name: float(row[col_a.name]), col_b.name: float(row[col_b.name])}
            for _, row in pair_df.iterrows()
        ]
        payloads.append(ChartPayload(
            chart_type=ChartType.SCATTER,
            title=f"Scatter – {col_a.name} vs {col_b.name} (r={corr_str})",
            field_placement=FieldPlacement(axis_x=col_a.name, axis_y=col_b.name),
            data=scatter_data,
            summary_text=f"Pearson correlation r = {corr_str}.",
        ))
        scatter_count += 1

    return payloads


# ---------------------------------------------------------------------------
# Tier 3 – Multivariate Analysis
# ---------------------------------------------------------------------------


def generate_multivariate(
    df: pd.DataFrame,
    columns: List[ColumnMeta],
) -> List[ChartPayload]:
    """Generate Tier-3 multivariate chart payloads.

    * 3+ NUMERIC → Bubble Chart (X, Y, Size; first TEXT as Color).
    * 2+ TEXT + 1+ NUMERIC → Pivot / Matrix Table.
    * DATE + NUMERIC + TEXT → Stacked Bar/Area Chart.
    """
    payloads: List[ChartPayload] = []

    date_cols = [c for c in columns if c.dtype_class == "DATE"]
    num_cols = [c for c in columns if c.dtype_class == "NUMERIC"]
    text_cols = [c for c in columns if c.dtype_class in ("TEXT", "GEO")]

    # ── Bubble Chart (3+ NUMERIC) ──────────────────────────────
    if len(num_cols) >= 3:
        x_col, y_col, size_col = num_cols[0], num_cols[1], num_cols[2]
        color_col = text_cols[0] if text_cols else None

        cols_needed = [x_col.name, y_col.name, size_col.name]
        bubble_df = df[cols_needed].dropna()
        if not bubble_df.empty:
            bubble_data = []
            for idx, row in bubble_df.iterrows():
                d: Dict[str, Any] = {
                    x_col.name: float(row[x_col.name]),
                    y_col.name: float(row[y_col.name]),
                    size_col.name: float(row[size_col.name]),
                }
                if color_col is not None:
                    d[color_col.name] = str(df.at[idx, color_col.name])
                bubble_data.append(d)

            payloads.append(ChartPayload(
                chart_type=ChartType.BUBBLE,
                title=f"Bubble – {x_col.name} × {y_col.name} (size: {size_col.name})",
                field_placement=FieldPlacement(
                    axis_x=x_col.name,
                    axis_y=y_col.name,
                    size=size_col.name,
                    color=color_col.name if color_col else None,
                ),
                data=bubble_data,
                summary_text=f"4D bubble chart: {x_col.name}, {y_col.name}, {size_col.name}"
                    + (f", colored by {color_col.name}" if color_col else "") + ".",
            ))

    # ── Pivot / Matrix Table (2+ TEXT, 1+ NUMERIC) ─────────────
    if len(text_cols) >= 2 and len(num_cols) >= 1:
        row_col, col_col, val_col = text_cols[0], text_cols[1], num_cols[0]
        try:
            pivot = pd.pivot_table(
                df, index=row_col.name, columns=col_col.name,
                values=val_col.name, aggfunc="sum", fill_value=0,
            )
            pivot = pivot.iloc[:50, :30]
            # Convert to list-of-dicts format
            matrix_data = []
            for idx_val in pivot.index:
                row_dict: Dict[str, Any] = {row_col.name: str(idx_val)}
                for c in pivot.columns:
                    row_dict[str(c)] = float(pivot.at[idx_val, c])
                matrix_data.append(row_dict)

            payloads.append(ChartPayload(
                chart_type=ChartType.MATRIX,
                title=f"Matrix – {val_col.name} by {row_col.name} × {col_col.name}",
                field_placement=FieldPlacement(
                    rows=row_col.name, columns=col_col.name, values=val_col.name,
                ),
                data=matrix_data,
                summary_text=(
                    f"Pivot: sum of {val_col.name} by {row_col.name} × {col_col.name}."
                ),
            ))
        except Exception:
            logger.warning("Pivot table generation failed; skipping.", exc_info=True)

    # ── Stacked Bar/Area (DATE + NUMERIC + TEXT) ───────────────
    if date_cols and num_cols and text_cols:
        d_meta, n_meta, t_meta = date_cols[0], num_cols[0], text_cols[0]
        temp = df[[d_meta.name, n_meta.name, t_meta.name]].copy()
        temp[d_meta.name] = pd.to_datetime(temp[d_meta.name], errors="coerce")
        temp = temp.dropna()

        if not temp.empty:
            top_cats = (
                temp.groupby(t_meta.name)[n_meta.name]
                .sum().nlargest(10).index.tolist()
            )
            temp = temp[temp[t_meta.name].isin(top_cats)]
            stacked = (
                temp.groupby([d_meta.name, t_meta.name])[n_meta.name]
                .sum().reset_index()
            ).sort_values(d_meta.name)

            stacked_data = [
                {
                    d_meta.name: str(row[d_meta.name]),
                    n_meta.name: float(row[n_meta.name]),
                    t_meta.name: str(row[t_meta.name]),
                }
                for _, row in stacked.iterrows()
            ]

            payloads.append(ChartPayload(
                chart_type=ChartType.STACKED_BAR,
                title=f"Stacked – {n_meta.name} over {d_meta.name} by {t_meta.name}",
                field_placement=FieldPlacement(
                    axis_x=d_meta.name, axis_y=n_meta.name, legend=t_meta.name,
                ),
                data=stacked_data,
                summary_text=(
                    f"Stacked {n_meta.name} over {d_meta.name}, "
                    f"segmented by top {len(top_cats)} {t_meta.name} values."
                ),
            ))

    return payloads


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class BaselineEDAResult(BaseModel):
    """Complete result from the automated baseline EDA pipeline."""

    univariate: List[ChartPayload] = Field(
        default_factory=list,
        description="Tier-1 univariate chart payloads (KPIs, histograms, bars).",
    )
    bivariate: List[ChartPayload] = Field(
        default_factory=list,
        description="Tier-2 bivariate chart payloads (lines, clustered bars, scatters).",
    )
    multivariate: List[ChartPayload] = Field(
        default_factory=list,
        description="Tier-3 multivariate chart payloads (bubbles, matrices, stacked).",
    )
    schema_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Column metadata and schema overview.",
    )
    row_count: int = Field(default=0, description="Total rows in the dataset.")
    column_count: int = Field(default=0, description="Total columns in the dataset.")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------


def run_baseline_eda(df: pd.DataFrame) -> BaselineEDAResult:
    """Run the complete automated baseline EDA and return all three tiers.

    Pipeline:
      1. Classify every column (TEXT / NUMERIC / DATE / GEO).
      2. Generate Tier-1 univariate payloads.
      3. Generate Tier-2 bivariate payloads.
      4. Generate Tier-3 multivariate payloads.
      5. Package everything into :class:`BaselineEDAResult`.
    """
    logger.info("Starting baseline EDA: %d rows × %d columns.", len(df), len(df.columns))

    col_metas = classify_columns(df)
    tier1 = generate_univariate(df, col_metas)
    tier2 = generate_bivariate(df, col_metas)
    tier3 = generate_multivariate(df, col_metas)

    schema_info: Dict[str, Any] = {
        "columns": [m.model_dump() for m in col_metas],
        "dtype_summary": {
            dtype: sum(1 for m in col_metas if m.dtype_class == dtype)
            for dtype in ("NUMERIC", "TEXT", "DATE", "GEO")
        },
    }

    logger.info(
        "Baseline EDA complete: %d univariate, %d bivariate, %d multivariate payloads.",
        len(tier1), len(tier2), len(tier3),
    )

    return BaselineEDAResult(
        univariate=tier1, bivariate=tier2, multivariate=tier3,
        schema_info=schema_info, row_count=len(df), column_count=len(df.columns),
    )
