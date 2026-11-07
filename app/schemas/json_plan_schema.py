"""Pydantic v2 models for the Structured JSON Query Plan (Page 2).

Defines the schema used to represent parsed natural-language queries as
structured, auditable execution plans.  Every query that arrives from the
LLM-powered NL → Plan translator is validated against these models before
the pandas / computation layer touches it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class FilterOp(str, Enum):
    """Whitelisted comparison operators for filter conditions."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="


class AggFn(str, Enum):
    """Whitelisted aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    DISTINCT_COUNT = "distinct_count"
    MIN = "min"
    MAX = "max"
    STD = "std"


class OperationType(str, Enum):
    """Top-level operation categories for a query plan."""

    AGGREGATION = "aggregation"
    FILTER = "filter"
    SORT = "sort"
    RAW = "raw"


class TimeIntelligence(str, Enum):
    """DAX-style time-intelligence modifiers."""

    TOTALYTD = "TOTALYTD"
    PREVIOUSMONTH = "PREVIOUSMONTH"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class FilterCondition(BaseModel):
    """A single column-level filter predicate.

    Attributes:
        col: Column name to filter on.
        op: Comparison operator (must be in the ``FilterOp`` whitelist).
        value: Literal value to compare against.
    """

    col: str = Field(..., min_length=1, description="Column name to filter on.")
    op: FilterOp = Field(..., description="Comparison operator.")
    value: Any = Field(..., description="Value to compare against.")

    @field_validator("op", mode="before")
    @classmethod
    def _validate_op(cls, v: Any) -> str:
        """Ensure the operator string is in the whitelist."""
        allowed = {e.value for e in FilterOp}
        if v not in allowed:
            raise ValueError(
                f"Invalid filter operator '{v}'. Allowed: {sorted(allowed)}"
            )
        return v


class AggregationSpec(BaseModel):
    """Specification for a single aggregation step.

    Attributes:
        col: Column to aggregate.
        fn: Aggregation function to apply (must be in the ``AggFn`` whitelist).
    """

    col: str = Field(..., min_length=1, description="Column to aggregate.")
    fn: AggFn = Field(..., description="Aggregation function.")

    @field_validator("fn", mode="before")
    @classmethod
    def _validate_fn(cls, v: Any) -> str:
        """Ensure the function name is in the whitelist."""
        allowed = {e.value for e in AggFn}
        if v not in allowed:
            raise ValueError(
                f"Invalid aggregation function '{v}'. Allowed: {sorted(allowed)}"
            )
        return v


class SortSpec(BaseModel):
    """Sort specification for the result set.

    Attributes:
        by: Column name to sort by.
        desc: ``True`` for descending order, ``False`` for ascending.
    """

    by: str = Field(..., min_length=1, description="Column to sort by.")
    desc: bool = Field(default=False, description="Descending order flag.")


# ---------------------------------------------------------------------------
# Top-level query plan
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """Structured JSON Query Plan — the canonical intermediate representation.

    Every natural-language question is first translated into this plan so that
    downstream computation is deterministic and auditable.

    Attributes:
        operation: High-level operation type.
        filter: Zero or more column-level filter predicates.
        group_by: Columns to group by (for aggregation operations).
        agg: Optional aggregation specification.
        sort: Optional sort specification.
        limit: Row limit for the result set (``None`` means no limit).
        time_intelligence: Optional DAX-style time modifier.
    """

    operation: OperationType = Field(
        ..., description="High-level operation type."
    )
    filter: List[FilterCondition] = Field(
        default_factory=list,
        description="Column-level filter predicates.",
    )
    group_by: List[str] = Field(
        default_factory=list,
        description="Columns to group by.",
    )
    agg: Optional[AggregationSpec] = Field(
        default=None,
        description="Aggregation specification.",
    )
    sort: Optional[SortSpec] = Field(
        default=None,
        description="Sort specification.",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum rows to return.",
    )
    time_intelligence: Optional[TimeIntelligence] = Field(
        default=None,
        description="DAX-style time-intelligence modifier.",
    )

    # -- cross-field validation ------------------------------------------------

    @model_validator(mode="after")
    def _aggregation_requires_agg_spec(self) -> "QueryPlan":
        """If operation is ``aggregation``, an ``agg`` spec must be present."""
        if self.operation == OperationType.AGGREGATION and self.agg is None:
            raise ValueError(
                "An 'agg' specification is required when operation is 'aggregation'."
            )
        return self


# ---------------------------------------------------------------------------
# Full query response envelope
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    """Envelope returned by the query execution pipeline.

    Attributes:
        plan: The validated query plan that was executed.
        data: Computed result rows.
        row_count: Number of rows in the result.
        columns: Column names present in the result.
        summary: Human-readable grounded summary of the result.
        error: Error message if execution failed (``None`` on success).
    """

    plan: QueryPlan = Field(..., description="Executed query plan.")
    data: List[dict[str, Any]] = Field(
        default_factory=list,
        description="Result rows as list of dicts.",
    )
    row_count: int = Field(default=0, ge=0, description="Number of result rows.")
    columns: List[str] = Field(
        default_factory=list,
        description="Column names in the result.",
    )
    summary: str = Field(
        default="",
        description="Grounded natural-language summary.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message on failure.",
    )
