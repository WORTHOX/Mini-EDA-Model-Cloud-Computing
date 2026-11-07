"""Natural Language → Structured JSON Query Plan Parser.

This module is the NLP heart of Page 2.  It converts free-form English questions
into validated :class:`QueryPlan` objects that the whitelisted Pandas executor
can process deterministically.

The parser is "pre-steered" via **Weight-Space Programming** (WSP):  at upload
time the ``WeightSpaceProgrammer`` synthesises ``DomainWeights`` — column
embeddings, domain vocabulary, query templates, and normalisation parameters —
which the parser loads at construction.  This is the practical equivalent of
injecting WSP-synthesised weight tensors ``[W1, W2, b1]`` into a compact model.

Parsing Pipeline
----------------
1. **Intent Classification** — keyword/pattern matching against a ranked intent
   vocabulary (aggregation, filter, sort, comparison, trend, distribution,
   correlation).
2. **Column Resolution** — fuzzy-matches mentioned terms to DataFrame column
   names using the WSP column embeddings (aliases + bigram Jaccard).
3. **Filter Extraction** — detects quoted values, comparison phrases, and named
   filter conditions (e.g. ``"completed orders"`` → Status == "Completed").
4. **Aggregation Detection** — maps keywords (total, average, count, …) to
   whitelisted ``AggFn`` values.
5. **Group-By Detection** — detects "by <column>" and "per <column>" phrases.
6. **Sort & Limit** — detects "top N" / "bottom N" constructs.
7. **Time Intelligence** — detects "year to date" / "YTD" / "previous month".

Test Cases (input → expected output intent)
-------------------------------------------
# "What is the total revenue?"                  → aggregation, agg=sum(Revenue)
# "Show me average sales by region"             → aggregation, agg=mean(Sales), group_by=[Region]
# "Top 10 products by revenue"                  → sort desc + limit 10
# "How many completed orders?"                  → aggregation, agg=count, filter Status==Completed
# "Revenue trend over time"                     → trend, agg=sum(Revenue), group_by date
# "Compare profit across departments"           → category_comparison, group_by=[Department]
# "What is the distribution of prices?"         → distribution
# "Year to date total sales"                    → aggregation + TOTALYTD
# "Bottom 5 customers by order count"           → sort asc + limit 5
# "Correlation between price and quantity"      → correlation
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.weight_programmer import DomainWeights, SchemaContext
from app.schemas.json_plan_schema import (
    AggFn,
    AggregationSpec,
    FilterCondition,
    FilterOp,
    OperationType,
    QueryPlan,
    SortSpec,
    TimeIntelligence,
)


# ---------------------------------------------------------------------------
# Keyword vocabularies
# ---------------------------------------------------------------------------

_AGG_KEYWORDS: Dict[str, str] = {
    "total": "sum",
    "sum": "sum",
    "add up": "sum",
    "combined": "sum",
    "overall": "sum",
    "aggregate": "sum",
    "average": "mean",
    "mean": "mean",
    "avg": "mean",
    "count": "count",
    "how many": "count",
    "number of": "count",
    "quantity of": "count",
    "unique": "distinct_count",
    "distinct": "distinct_count",
    "different": "distinct_count",
    "minimum": "min",
    "lowest": "min",
    "smallest": "min",
    "min": "min",
    "least": "min",
    "maximum": "max",
    "highest": "max",
    "largest": "max",
    "max": "max",
    "greatest": "max",
    "most": "max",
    "standard deviation": "std",
    "std": "std",
    "stdev": "std",
    "volatility": "std",
    "variance": "std",
}

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "trend": [
        "trend", "over time", "monthly", "yearly", "weekly", "daily",
        "growth", "change over", "evolution", "progression", "timeline",
        "time series", "by month", "by year", "by week", "by day",
        "month over month", "year over year",
    ],
    "comparison": [
        "compare", "vs", "versus", "difference", "between", "against",
        "comparison", "contrasting",
    ],
    "distribution": [
        "distribution", "spread", "range of", "histogram", "frequency",
        "dispersion", "breakdown of values",
    ],
    "correlation": [
        "correlation", "relationship", "associated", "relate",
        "co-relation", "dependency", "scatter",
    ],
    "proportion": [
        "proportion", "share", "percentage", "percent of", "fraction",
        "pie", "donut", "composition", "makeup", "breakdown by",
    ],
}

_GROUP_BY_SIGNALS: List[str] = [
    "by", "per", "for each", "grouped by", "across", "segmented by",
    "broken down by", "split by", "categorized by",
]

_FILTER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"where\s+(\w[\w\s]*?)\s+(is|are|equals?|=)\s+['\"]?(.+?)['\"]?(?:\s|$)", re.I), "=="),
    (re.compile(r"(?:only|just|for)\s+['\"]?(\w[\w\s]*?)['\"]?\s+(\w[\w\s]*?)(?:\s|$)", re.I), "=="),
    (re.compile(r"(\w+)\s*(?:>|greater than|more than|above|over)\s*(\d+\.?\d*)", re.I), ">"),
    (re.compile(r"(\w+)\s*(?:<|less than|fewer than|below|under)\s*(\d+\.?\d*)", re.I), "<"),
    (re.compile(r"(\w+)\s*(?:>=|at least)\s*(\d+\.?\d*)", re.I), ">="),
    (re.compile(r"(\w+)\s*(?:<=|at most)\s*(\d+\.?\d*)", re.I), "<="),
]

_SORT_TOP_RE = re.compile(r"(?:top|first|best|highest)\s+(\d+)", re.I)
_SORT_BOTTOM_RE = re.compile(r"(?:bottom|last|worst|lowest)\s+(\d+)", re.I)
_LIMIT_RE = re.compile(r"(?:limit|show|display)\s+(\d+)", re.I)

_TIME_INTEL_KEYWORDS: Dict[str, str] = {
    "year to date": "TOTALYTD",
    "ytd": "TOTALYTD",
    "year-to-date": "TOTALYTD",
    "cumulative this year": "TOTALYTD",
    "previous month": "PREVIOUSMONTH",
    "last month": "PREVIOUSMONTH",
    "prior month": "PREVIOUSMONTH",
}


# ---------------------------------------------------------------------------
# NL Query Parser
# ---------------------------------------------------------------------------

class NLQueryParser:
    """Natural-language → JSON Query Plan parser, pre-steered by WSP weights.

    At construction the parser receives ``DomainWeights`` synthesised by the
    Weight-Space Programmer.  These weights provide:

    * **Column embeddings** with alias sets for fuzzy column resolution.
    * **Domain vocabulary** for recognising dataset-specific terms.
    * **Query templates** for matching common analytical patterns.
    * **Normalisation parameters** for value-range reasoning.

    Usage::

        parser = NLQueryParser(domain_weights)
        plan = parser.parse("What is the total revenue by region?", schema_ctx)
    """

    def __init__(self, domain_weights: DomainWeights) -> None:
        self._weights = domain_weights
        # Build a flat alias → column_name lookup for fast resolution.
        self._alias_map: Dict[str, str] = {}
        for col, emb in domain_weights.column_embeddings.items():
            for alias in emb.get("aliases", set()):
                self._alias_map[alias.lower()] = col
            # Also map the canonical name itself.
            self._alias_map[col.lower()] = col

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, query: str, schema_context: SchemaContext) -> QueryPlan:
        """Parse a natural-language question into a :class:`QueryPlan`.

        Args:
            query: The raw user question string.
            schema_context: The current dataset's schema context.

        Returns:
            A validated ``QueryPlan`` ready for execution.

        Raises:
            ValueError: If the query cannot be meaningfully parsed.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        q = query.strip()
        q_lower = q.lower()

        # ---- Step 1: Detect time intelligence ----
        time_intel = self._detect_time_intelligence(q_lower)

        # ---- Step 2: Detect sort & limit ----
        sort_desc, limit = self._detect_sort_limit(q_lower)
        sort_spec: Optional[SortSpec] = None

        # ---- Step 3: Detect aggregation function ----
        agg_fn = self._detect_aggregation(q_lower)

        # ---- Step 4: Resolve columns mentioned in the query ----
        resolved_columns = self._resolve_columns(q_lower, schema_context)

        # ---- Step 5: Detect group-by columns ----
        group_by = self._detect_group_by(q_lower, schema_context, resolved_columns)

        # ---- Step 6: Detect filters ----
        filters = self._detect_filters(q_lower, schema_context)

        # ---- Step 7: Determine aggregation column ----
        agg_col = self._pick_agg_column(resolved_columns, schema_context, agg_fn)

        # ---- Step 8: Build sort spec if sort was detected ----
        if sort_desc is not None:
            sort_col = None
            if agg_col:
                sort_col = agg_col
            else:
                # Pick first numeric column from resolved
                for rc in resolved_columns:
                    if rc in schema_context.numeric_columns:
                        sort_col = rc
                        break
            if sort_col:
                sort_spec = SortSpec(by=sort_col, desc=sort_desc)

        # ---- Step 9: Classify intent ----
        intent = self._classify_intent(q_lower, agg_fn, group_by, schema_context)

        # ---- Step 9b: Trend auto-inference ----
        # If the query has trend keywords but no agg_fn was detected,
        # auto-infer SUM on the first numeric column and group by date.
        if intent == "trend" and not agg_fn:
            agg_fn = "sum"
            agg_col = self._pick_agg_column(resolved_columns, schema_context, agg_fn)
            # Group by the first date column if not already grouped by one.
            if schema_context.date_columns and not any(
                g in schema_context.date_columns for g in group_by
            ):
                group_by = [schema_context.date_columns[0]] + [
                    g for g in group_by if g not in schema_context.date_columns
                ]

        # ---- Step 9c: Comparison / correlation auto-inference ----
        # "profit vs cost", "X vs Y" → correlation / scatter, group_by both columns
        if intent in ("comparison", "correlation") and not agg_fn:
            # Resolved columns contain the two compared columns — no agg needed,
            # treat as RAW (the executor returns all rows with those columns).
            # We keep operation=RAW so the scatter chart has all data points.
            pass

        # ---- Step 9d: Distribution auto-inference ----
        # "distribution of Revenue" → histogram; infer COUNT aggregation with no group
        if intent == "distribution" and not agg_fn:
            agg_fn = "count"
            agg_col = self._pick_agg_column(resolved_columns, schema_context, agg_fn)

        # ---- Step 10: Determine operation type ----
        if agg_fn and agg_col:
            operation = OperationType.AGGREGATION
        elif sort_spec:
            operation = OperationType.SORT
        elif filters:
            operation = OperationType.FILTER
        else:
            operation = OperationType.RAW

        # ---- Step 11: Build aggregation spec ----
        agg_spec: Optional[AggregationSpec] = None
        if agg_fn and agg_col:
            agg_spec = AggregationSpec(col=agg_col, fn=agg_fn)

        # ---- Step 12: If sort without explicit agg, ensure we have agg ----
        if sort_spec and not agg_spec:
            # "top 10 by revenue" implies aggregation
            if group_by:
                agg_spec = AggregationSpec(col=sort_spec.by, fn="sum")
                operation = OperationType.AGGREGATION

        # ---- Step 13: Build the plan ----
        plan = QueryPlan(
            operation=operation,
            filter=filters,
            group_by=group_by,
            agg=agg_spec,
            sort=sort_spec,
            limit=limit,
            time_intelligence=time_intel,
        )

        return plan

    # ------------------------------------------------------------------ #
    # Column resolution
    # ------------------------------------------------------------------ #

    def _resolve_columns(
        self, q_lower: str, ctx: SchemaContext
    ) -> List[str]:
        """Find all dataset columns mentioned in the query.

        Uses three strategies:
        1. Exact alias match against the WSP column embeddings.
        2. Fuzzy match using ``difflib.get_close_matches``.
        3. Token overlap with column name parts.
        """
        found: List[str] = []
        found_set: set = set()

        # Strategy 1: Exact alias match (longest-match-first to avoid partial hits).
        sorted_aliases = sorted(self._alias_map.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in q_lower and self._alias_map[alias] not in found_set:
                col = self._alias_map[alias]
                found.append(col)
                found_set.add(col)

        # Strategy 2: Fuzzy match on remaining tokens.
        tokens = re.findall(r"\b[a-zA-Z_]\w*\b", q_lower)
        all_col_names = [c.lower() for c in ctx.column_names]
        for token in tokens:
            if len(token) < 3:
                continue
            matches = difflib.get_close_matches(
                token, all_col_names, n=1, cutoff=0.7
            )
            if matches:
                # Map back to original case.
                idx = all_col_names.index(matches[0])
                original = ctx.column_names[idx]
                if original not in found_set:
                    found.append(original)
                    found_set.add(original)

        return found

    def _fuzzy_match_column(
        self, term: str, columns: List[str]
    ) -> Optional[str]:
        """Find the closest matching column name for a term."""
        lower_cols = [c.lower() for c in columns]
        matches = difflib.get_close_matches(
            term.lower(), lower_cols, n=1, cutoff=0.6
        )
        if matches:
            idx = lower_cols.index(matches[0])
            return columns[idx]

        # Fall back to alias map.
        return self._alias_map.get(term.lower())

    # ------------------------------------------------------------------ #
    # Intent classification
    # ------------------------------------------------------------------ #

    def _classify_intent(
        self,
        q_lower: str,
        agg_fn: Optional[str],
        group_by: List[str],
        ctx: SchemaContext,
    ) -> str:
        """Classify the analytical intent of the query."""
        # Check explicit intent keywords.
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in q_lower:
                    return intent

        # Infer from structure.
        if agg_fn and not group_by:
            return "single_value"
        if agg_fn and group_by:
            # Check if group_by is a date column → trend.
            for g in group_by:
                if g in ctx.date_columns:
                    return "trend"
            return "category_comparison"
        if any(kw in q_lower for kw in ("top", "bottom", "rank", "highest", "lowest")):
            return "sort"

        return "raw_table"

    # ------------------------------------------------------------------ #
    # Aggregation detection
    # ------------------------------------------------------------------ #

    def _detect_aggregation(self, q_lower: str) -> Optional[str]:
        """Detect the aggregation function from query keywords.

        Uses longest-match-first to avoid "min" matching inside "minimum".
        """
        sorted_keywords = sorted(_AGG_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in q_lower:
                return _AGG_KEYWORDS[keyword]
        return None

    def _pick_agg_column(
        self,
        resolved_columns: List[str],
        ctx: SchemaContext,
        agg_fn: Optional[str],
    ) -> Optional[str]:
        """Pick the best aggregation target column.

        Prefers numeric columns.  If the aggregation function is 'count' or
        'distinct_count', any column type works.
        """
        if not agg_fn:
            return None

        # For count, prefer the first resolved column, or any column.
        if agg_fn in ("count", "distinct_count"):
            for col in resolved_columns:
                return col
            # Fall back to first column in schema.
            return ctx.column_names[0] if ctx.column_names else None

        # For numeric aggs, find a numeric column among resolved ones.
        for col in resolved_columns:
            if col in ctx.numeric_columns:
                return col

        # Fall back to the first numeric column in the schema.
        if ctx.numeric_columns:
            return ctx.numeric_columns[0]

        return None

    # ------------------------------------------------------------------ #
    # Group-by detection
    # ------------------------------------------------------------------ #

    def _detect_group_by(
        self,
        q_lower: str,
        ctx: SchemaContext,
        resolved_columns: List[str],
    ) -> List[str]:
        """Detect group-by columns from 'by <col>', 'per <col>', etc."""
        group_cols: List[str] = []
        group_set: set = set()

        for signal in _GROUP_BY_SIGNALS:
            # Allow column term to run to end of string OR to a stop word.
            pattern = re.compile(
                rf"\b{re.escape(signal)}\s+(\w[\w\s]*?)(?:\s+(?:and|,|where|with|for|in|the)|$)",
                re.I,
            )
            match = pattern.search(q_lower)
            if match:
                candidate = match.group(1).strip()
                col = self._fuzzy_match_column(candidate, ctx.column_names)
                if col:
                    # If matched column is numeric and there is a categorical/date/geo column
                    # among the resolved columns (e.g. "top 5 products by profit"),
                    # the numeric column is the metric, NOT the group dimension!
                    if col in ctx.numeric_columns:
                        cat_candidates = [
                            c for c in resolved_columns
                            if c in ctx.categorical_columns or c in ctx.date_columns or c in ctx.geo_columns
                        ]
                        if cat_candidates:
                            for cat_col in cat_candidates:
                                if cat_col not in group_set:
                                    group_cols.append(cat_col)
                                    group_set.add(cat_col)
                            continue

                    if col not in group_set:
                        group_cols.append(col)
                        group_set.add(col)

        # If no explicit group-by found but we have resolved categorical columns
        # and more than one column was resolved, use the first categorical as group.
        if not group_cols and len(resolved_columns) > 1:
            for rc in resolved_columns:
                if rc in ctx.categorical_columns and rc not in group_set:
                    group_cols.append(rc)
                    group_set.add(rc)
                    break

        return group_cols

    # ------------------------------------------------------------------ #
    # Filter detection
    # ------------------------------------------------------------------ #

    def _detect_filters(
        self, q_lower: str, ctx: SchemaContext
    ) -> List[FilterCondition]:
        """Extract filter conditions from the query text.

        Handles:
          - "where <col> is <value>"
          - "only <value> <col>"  / "for <value> <col>"
          - "<col> > <number>", "<col> < <number>", etc.
          - Quoted values: 'Completed', "Active"
        """
        filters: List[FilterCondition] = []
        seen: set = set()

        # Explicit quoted value detection.
        quoted_re = re.compile(
            r"(?:where|for|with|only)\s+(\w[\w\s]*?)\s*(?:is|=|==|equals?)\s*['\"](.+?)['\"]",
            re.I,
        )
        for m in quoted_re.finditer(q_lower):
            col_hint = m.group(1).strip()
            value = m.group(2).strip()
            col = self._fuzzy_match_column(col_hint, ctx.column_names)
            if col:
                key = (col, "==", value)
                if key not in seen:
                    filters.append(
                        FilterCondition(col=col, op=FilterOp.EQ, value=value)
                    )
                    seen.add(key)

        # Status/category shorthand: "completed orders" → Status == "Completed"
        # This uses domain vocabulary to detect categorical values in the query.
        for col in ctx.categorical_columns:
            samples = ctx.sample_values.get(col, [])
            for sample_val in samples:
                sv = str(sample_val).lower()
                if len(sv) > 2 and sv in q_lower:
                    key = (col, "==", str(sample_val))
                    if key not in seen:
                        filters.append(
                            FilterCondition(
                                col=col,
                                op=FilterOp.EQ,
                                value=str(sample_val),
                            )
                        )
                        seen.add(key)

        # Numeric comparison filters.
        numeric_ops = [
            (re.compile(rf"(\w+)\s*(?:greater than|more than|above|over|>)\s*(\d+\.?\d*)", re.I), FilterOp.GT),
            (re.compile(rf"(\w+)\s*(?:less than|fewer than|below|under|<)\s*(\d+\.?\d*)", re.I), FilterOp.LT),
            (re.compile(rf"(\w+)\s*(?:at least|>=)\s*(\d+\.?\d*)", re.I), FilterOp.GE),
            (re.compile(rf"(\w+)\s*(?:at most|<=)\s*(\d+\.?\d*)", re.I), FilterOp.LE),
        ]
        for pattern, op in numeric_ops:
            for m in pattern.finditer(q_lower):
                col_hint = m.group(1).strip()
                num_val = self._extract_numeric_value(m.group(2))
                col = self._fuzzy_match_column(col_hint, ctx.column_names)
                if col and num_val is not None:
                    key = (col, op.value, num_val)
                    if key not in seen:
                        filters.append(
                            FilterCondition(col=col, op=op, value=num_val)
                        )
                        seen.add(key)

        return filters

    # ------------------------------------------------------------------ #
    # Sort & limit detection
    # ------------------------------------------------------------------ #

    def _detect_sort_limit(
        self, q_lower: str
    ) -> Tuple[Optional[bool], Optional[int]]:
        """Detect 'top N', 'bottom N', or 'limit N' patterns."""
        sort_desc: Optional[bool] = None
        limit: Optional[int] = None

        # "top 10 ..."
        m = _SORT_TOP_RE.search(q_lower)
        if m:
            limit = int(m.group(1))
            sort_desc = True

        # "bottom 5 ..."
        m = _SORT_BOTTOM_RE.search(q_lower)
        if m:
            limit = int(m.group(1))
            sort_desc = False

        # Explicit "limit N"
        if limit is None:
            m = _LIMIT_RE.search(q_lower)
            if m:
                limit = int(m.group(1))

        # Build sort spec only if we know the direction (by= resolved later)
        sort_spec = None
        if sort_desc is not None:
            # Placeholder — by= will be resolved in parse() once columns are known
            sort_spec = sort_desc  # Store as bool, convert to SortSpec later

        return sort_spec, limit

    # ------------------------------------------------------------------ #
    # Time intelligence detection
    # ------------------------------------------------------------------ #

    def _detect_time_intelligence(self, q_lower: str) -> Optional[str]:
        """Detect DAX-style time-intelligence keywords."""
        # Check longest phrases first.
        for phrase in sorted(_TIME_INTEL_KEYWORDS.keys(), key=len, reverse=True):
            if phrase in q_lower:
                return _TIME_INTEL_KEYWORDS[phrase]
        return None

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_numeric_value(text: str) -> Optional[float]:
        """Extract a numeric value from text."""
        cleaned = re.sub(r"[,$%]", "", text.strip())
        try:
            return float(cleaned)
        except ValueError:
            return None

