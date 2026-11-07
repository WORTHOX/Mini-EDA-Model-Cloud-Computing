"""Field Placement Memory Rules Engine.

Maps column metadata (name, dtype classification, role) to visual encoding
channels based on strict, deterministic placement rules.  The rules mirror
Power BI / Tableau best-practice conventions:

    1. Text / Categorical → Axis · Legend · Group
    2. Numeric / Quantitative → Values · Y-Axis · Size
    3. Date / Time → Time-Axis · X-Axis
    4. Geographic → Location · Color
    5. Secondary Information → Tooltip
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from app.schemas.visual_payload import ChartType, FieldPlacement


# ---------------------------------------------------------------------------
# Column metadata models
# ---------------------------------------------------------------------------

class DtypeClass(str, Enum):
    """High-level data-type classification for a column."""

    TEXT = "TEXT"
    NUMERIC = "NUMERIC"
    DATE = "DATE"
    GEO = "GEO"


class ColumnRole(str, Enum):
    """Semantic role a column plays in the analysis."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    MEASURE = "MEASURE"
    DIMENSION = "DIMENSION"


class ColumnMeta(BaseModel):
    """Metadata descriptor for a single DataFrame column.

    Attributes:
        name: The column name as it appears in the DataFrame.
        dtype_class: High-level type bucket (TEXT, NUMERIC, DATE, GEO).
        role: Semantic role (PRIMARY, SECONDARY, MEASURE, DIMENSION).
    """

    name: str = Field(..., min_length=1, description="Column name.")
    dtype_class: DtypeClass = Field(..., description="Data-type classification.")
    role: ColumnRole = Field(..., description="Semantic role of the column.")


# ---------------------------------------------------------------------------
# Chart-family helpers
# ---------------------------------------------------------------------------

_CARTESIAN_CHARTS = {
    ChartType.BAR,
    ChartType.COLUMN,
    ChartType.LINE,
    ChartType.AREA,
    ChartType.SCATTER,
    ChartType.BUBBLE,
    ChartType.HISTOGRAM,
    ChartType.WATERFALL,
    ChartType.STACKED_BAR,
    ChartType.STACKED_COLUMN,
    ChartType.CLUSTERED_BAR,
    ChartType.CLUSTERED_COLUMN,
}

_PIE_FAMILY = {ChartType.PIE, ChartType.DONUT}

_MATRIX_FAMILY = {ChartType.MATRIX, ChartType.TREEMAP}

_CARD_FAMILY = {ChartType.CARD, ChartType.MULTI_ROW_CARD}


# ---------------------------------------------------------------------------
# Internal bucket helpers
# ---------------------------------------------------------------------------

def _bucket_columns(
    columns: List[ColumnMeta],
) -> tuple[
    list[ColumnMeta],
    list[ColumnMeta],
    list[ColumnMeta],
    list[ColumnMeta],
]:
    """Partition columns into (texts, numerics, dates, geos)."""
    texts: list[ColumnMeta] = []
    numerics: list[ColumnMeta] = []
    dates: list[ColumnMeta] = []
    geos: list[ColumnMeta] = []

    for col in columns:
        if col.dtype_class == DtypeClass.TEXT:
            texts.append(col)
        elif col.dtype_class == DtypeClass.NUMERIC:
            numerics.append(col)
        elif col.dtype_class == DtypeClass.DATE:
            dates.append(col)
        elif col.dtype_class == DtypeClass.GEO:
            geos.append(col)

    return texts, numerics, dates, geos


def _pick_primary(cols: list[ColumnMeta]) -> ColumnMeta | None:
    """Return the first PRIMARY-role column, or the first column."""
    for c in cols:
        if c.role == ColumnRole.PRIMARY:
            return c
    return cols[0] if cols else None


def _pick_secondary(cols: list[ColumnMeta]) -> list[ColumnMeta]:
    """Return all SECONDARY-role columns."""
    return [c for c in cols if c.role == ColumnRole.SECONDARY]


def _names(cols: list[ColumnMeta]) -> list[str]:
    """Extract column names."""
    return [c.name for c in cols]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_fields(
    columns: List[ColumnMeta],
    chart_type: ChartType,
) -> FieldPlacement:
    """Map column metadata to visual encoding channels.

    Applies the Field Placement Memory Rules to deterministically assign
    each column to the appropriate visual channel based on its data type
    classification and semantic role.

    Args:
        columns: Ordered list of column metadata descriptors.
        chart_type: The target chart type to map fields for.

    Returns:
        A fully populated ``FieldPlacement`` ready for the chart payload.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError("Cannot map fields for an empty column list.")

    texts, numerics, dates, geos = _bucket_columns(columns)

    # Collect tooltip candidates from secondary columns across all types
    tooltip_cols: list[str] = []
    for bucket in (texts, numerics, dates, geos):
        tooltip_cols.extend(c.name for c in _pick_secondary(bucket))

    placement = FieldPlacement()

    # ----- CARD family: just values, no axes -----
    if chart_type in _CARD_FAMILY:
        if numerics:
            placement.values = (
                numerics[0].name
                if len(numerics) == 1
                else _names(numerics)
            )
        if tooltip_cols:
            placement.tooltip = tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        return placement

    # ----- MAP: geographic focus -----
    if chart_type == ChartType.MAP:
        primary_geo = _pick_primary(geos)
        if primary_geo:
            placement.location = primary_geo.name
        primary_numeric = _pick_primary(numerics)
        if primary_numeric:
            placement.color = primary_numeric.name
        if len(numerics) > 1:
            placement.values = numerics[1].name
        if tooltip_cols:
            placement.tooltip = tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        return placement

    # ----- GAUGE: single numeric value -----
    if chart_type == ChartType.GAUGE:
        primary_numeric = _pick_primary(numerics)
        if primary_numeric:
            placement.values = primary_numeric.name
        if tooltip_cols:
            placement.tooltip = tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        return placement

    # ----- PIE / DONUT family -----
    if chart_type in _PIE_FAMILY:
        primary_text = _pick_primary(texts)
        primary_numeric = _pick_primary(numerics)
        if primary_text:
            placement.legend = primary_text.name
        if primary_numeric:
            placement.values = primary_numeric.name
        if tooltip_cols:
            placement.tooltip = tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        return placement

    # ----- MATRIX / TREEMAP family -----
    if chart_type in _MATRIX_FAMILY:
        # Texts → rows/columns for hierarchical breakdown
        if texts:
            placement.rows = texts[0].name
            if len(texts) > 1:
                placement.columns = texts[1].name
        primary_numeric = _pick_primary(numerics)
        if primary_numeric:
            placement.values = primary_numeric.name
        if geos:
            placement.color = geos[0].name
        if tooltip_cols:
            placement.tooltip = tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        return placement

    # ----- CARTESIAN charts (bar, line, scatter, etc.) -----
    if chart_type in _CARTESIAN_CHARTS:
        return _map_cartesian(
            chart_type, texts, numerics, dates, geos, tooltip_cols
        )

    # Fallback — best effort for unknown chart types
    return _map_cartesian(
        chart_type, texts, numerics, dates, geos, tooltip_cols
    )


def _map_cartesian(
    chart_type: ChartType,
    texts: list[ColumnMeta],
    numerics: list[ColumnMeta],
    dates: list[ColumnMeta],
    geos: list[ColumnMeta],
    tooltip_cols: list[str],
) -> FieldPlacement:
    """Apply placement rules for Cartesian (X/Y axis) charts.

    Rule priority:
        1. Date → X-axis (time axis)
        2. Text/Categorical → X-axis (category axis) or Legend
        3. Numeric → Y-axis / Values / Size
        4. Geographic → Color
        5. Secondary → Tooltip
    """
    placement = FieldPlacement()

    # --- X-axis assignment ---
    # Rule 3: Date columns get priority on X-axis
    if dates:
        primary_date = _pick_primary(dates)
        placement.axis_x = primary_date.name if primary_date else dates[0].name
    elif texts:
        # Rule 1: Text/categorical → axis
        primary_text = _pick_primary(texts)
        placement.axis_x = primary_text.name if primary_text else texts[0].name

    # --- Y-axis assignment ---
    # Rule 2: Numeric → Y-axis
    primary_numeric = _pick_primary(numerics)
    if primary_numeric:
        placement.axis_y = primary_numeric.name
    elif numerics:
        placement.axis_y = numerics[0].name

    # --- Values for multi-value charts ---
    if len(numerics) > 1:
        remaining_numerics = [
            n.name for n in numerics if n.name != placement.axis_y
        ]
        if remaining_numerics:
            placement.values = (
                remaining_numerics[0]
                if len(remaining_numerics) == 1
                else remaining_numerics
            )

    # --- Legend ---
    # If we used a date on X and have text columns, use text as legend
    if dates and texts:
        primary_text = _pick_primary(texts)
        placement.legend = primary_text.name if primary_text else texts[0].name
    # Extra text columns beyond the first become legend
    elif texts and len(texts) > 1:
        placement.legend = texts[1].name

    # --- Size (bubble charts) ---
    if chart_type == ChartType.BUBBLE and len(numerics) >= 3:
        # Third numeric → size
        size_candidates = [
            n.name
            for n in numerics
            if n.name != placement.axis_y
            and n.name
            not in (
                placement.values
                if isinstance(placement.values, list)
                else [placement.values]
                if placement.values
                else []
            )
        ]
        if size_candidates:
            placement.size = size_candidates[0]

    # --- Geographic → color ---
    if geos:
        placement.color = geos[0].name

    # --- Tooltip ---
    if tooltip_cols:
        placement.tooltip = (
            tooltip_cols[0] if len(tooltip_cols) == 1 else tooltip_cols
        )

    return placement
