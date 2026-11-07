"""Pydantic models for chart / visual payloads sent to the frontend.

Every visualisation that the backend produces is serialised as a
``ChartPayload`` — a self-contained envelope that the frontend renderer
(Plotly, ECharts, or a custom React layer) can consume without additional
server round-trips.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chart type catalogue — company-approved visuals
# ---------------------------------------------------------------------------

class ChartType(str, Enum):
    """Exhaustive catalogue of company-approved chart types.

    Every visual the system produces must be one of these.  The frontend
    renderer uses the value to choose the correct Plotly / ECharts trace type.
    """

    BAR = "BAR"
    COLUMN = "COLUMN"
    LINE = "LINE"
    AREA = "AREA"
    SCATTER = "SCATTER"
    BUBBLE = "BUBBLE"
    PIE = "PIE"
    DONUT = "DONUT"
    HISTOGRAM = "HISTOGRAM"
    MATRIX = "MATRIX"
    TREEMAP = "TREEMAP"
    WATERFALL = "WATERFALL"
    GAUGE = "GAUGE"
    MAP = "MAP"
    CARD = "CARD"
    MULTI_ROW_CARD = "MULTI_ROW_CARD"
    STACKED_BAR = "STACKED_BAR"
    STACKED_COLUMN = "STACKED_COLUMN"
    CLUSTERED_BAR = "CLUSTERED_BAR"
    CLUSTERED_COLUMN = "CLUSTERED_COLUMN"


# ---------------------------------------------------------------------------
# Field placement — where each data column lands on the visual
# ---------------------------------------------------------------------------

class FieldPlacement(BaseModel):
    """Describes how data columns map to visual encoding channels.

    Each attribute corresponds to an encoding channel on the chart.
    The values are column names (or lists of column names) from the result
    DataFrame.

    Attributes:
        axis_x: Column(s) mapped to the X-axis.
        axis_y: Column(s) mapped to the Y-axis.
        values: Column(s) used as value encodings (e.g. pie slices).
        legend: Column used for the legend / series split.
        size: Column controlling marker size (bubbles, etc.).
        color: Column controlling colour encoding.
        tooltip: Column(s) shown on hover tooltip.
        location: Column with geographic identifiers (country, state, lat/lon).
        rows: Column(s) for small-multiples row facets (matrix visuals).
        columns: Column(s) for small-multiples column facets (matrix visuals).
    """

    axis_x: Optional[str | List[str]] = Field(
        default=None, description="X-axis column(s)."
    )
    axis_y: Optional[str | List[str]] = Field(
        default=None, description="Y-axis column(s)."
    )
    values: Optional[str | List[str]] = Field(
        default=None, description="Value column(s)."
    )
    legend: Optional[str] = Field(
        default=None, description="Legend / series-split column."
    )
    size: Optional[str] = Field(
        default=None, description="Size-encoding column."
    )
    color: Optional[str] = Field(
        default=None, description="Colour-encoding column."
    )
    tooltip: Optional[str | List[str]] = Field(
        default=None, description="Tooltip column(s)."
    )
    location: Optional[str] = Field(
        default=None, description="Geographic location column."
    )
    rows: Optional[str | List[str]] = Field(
        default=None, description="Row-facet column(s) for matrix visuals."
    )
    columns: Optional[str | List[str]] = Field(
        default=None, description="Column-facet column(s) for matrix visuals."
    )


# ---------------------------------------------------------------------------
# Chart payload — the full envelope sent to the frontend
# ---------------------------------------------------------------------------

class ChartPayload(BaseModel):
    """Self-contained chart payload for frontend rendering.

    This is the single contract between backend and frontend for every
    visualisation.

    Attributes:
        chart_type: The company-approved chart type to render.
        title: Human-readable chart title.
        field_placement: Mapping of data columns to visual channels.
        data: The actual data rows (list of flat dicts).
        summary_text: Grounded 1-2 line natural-language summary.
        config: Optional rendering configuration (Plotly layout overrides, etc.).
    """

    chart_type: ChartType = Field(
        ..., description="Chart type to render."
    )
    title: str = Field(
        ..., min_length=1, description="Chart title."
    )
    field_placement: FieldPlacement = Field(
        ..., description="Column → visual-channel mapping."
    )
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Result data rows.",
    )
    summary_text: str = Field(
        default="",
        description="Grounded natural-language summary of the chart.",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Plotly / renderer configuration overrides.",
    )
