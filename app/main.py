"""
Weight-Space Synthesized Natural-Language EDA Engine
=====================================================
FastAPI application entrypoint.

Exposes endpoints for:
  - CSV upload & automated baseline EDA (Page 1)
  - Natural language query & custom exploration (Page 2)
  - Health check & schema info

Architecture follows the SOP procedure: CSV → Schema Profiler → Page 1 / Page 2.
The Weight-Space Programming layer pre-steers the NLP parser at upload time.
"""

from __future__ import annotations

import io
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.executor import ExecutionResult, execute_query_plan
from app.core.field_mapper import ColumnMeta as FieldColumnMeta
from app.core.nlp_parser import NLQueryParser
from app.core.page1_baseline_eda import BaselineEDAResult, run_baseline_eda
from app.core.summary_engine import generate_summary
from app.core.visual_selector import infer_intent, select_visual
from app.core.weight_programmer import SchemaContext, WeightSpaceProgrammer
from app.schemas.json_plan_schema import QueryPlan
from app.schemas.visual_payload import ChartPayload, ChartType, FieldPlacement

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weight-Space EDA Engine",
    description=(
        "Domain-agnostic Exploratory Data Analysis & Interactive NL QA Engine. "
        "Powered by Weight-Space Programming (Takehi et al., 2026)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-Memory Session State
# ---------------------------------------------------------------------------

# Stores the currently loaded dataset and its derived artefacts.
# In a production deployment this would use Redis / session storage.
_session: Dict[str, Any] = {
    "df": None,
    "schema_context": None,
    "domain_weights": None,
    "parser": None,
    "baseline_eda": None,
    "filename": None,
}


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response returned after a successful CSV upload."""
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    message: str


class QueryRequest(BaseModel):
    """Page 2 natural-language query request."""
    question: str


class QueryResponse(BaseModel):
    """Page 2 query response containing chart + summary."""
    query_plan: Dict[str, Any]
    chart_payload: Dict[str, Any]
    summary: str
    intent: str


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    dataset_loaded: bool
    dataset_name: Optional[str]
    row_count: Optional[int]
    column_count: Optional[int]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    return FileResponse("app/static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint."""
    df = _session.get("df")
    return HealthResponse(
        status="ok",
        dataset_loaded=df is not None,
        dataset_name=_session.get("filename"),
        row_count=len(df) if df is not None else None,
        column_count=len(df.columns) if df is not None else None,
    )


def _process_and_store_dataset(df: pd.DataFrame, filename: str) -> UploadResponse:
    """Run WSP, initialize parser, execute baseline EDA, and store in session."""
    if df.empty:
        raise HTTPException(status_code=400, detail="The dataset is empty.")

    # ---- Weight-Space Programming: synthesise domain weights ----
    wsp = WeightSpaceProgrammer()
    schema_ctx = wsp.analyze_schema(df)
    domain_weights = wsp.synthesize_domain_weights(schema_ctx)

    # ---- Initialize NLP parser with synthesized weights ----
    parser = NLQueryParser(domain_weights)

    # ---- Run automated baseline EDA (Page 1) ----
    baseline = run_baseline_eda(df)

    # ---- Persist in session ----
    _session.update(
        {
            "df": df,
            "schema_context": schema_ctx,
            "domain_weights": domain_weights,
            "parser": parser,
            "baseline_eda": baseline,
            "filename": filename,
        }
    )

    return UploadResponse(
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        message="Dataset loaded successfully. Baseline EDA generated.",
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a cleaned CSV dataset.

    On upload the system:
      1. Loads the CSV into an in-memory DataFrame.
      2. Runs the Weight-Space Programmer to synthesize domain weights.
      3. Initializes the NLP parser with the synthesized weights.
      4. Runs the automated baseline EDA (Page 1).
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV: {exc}",
        )

    return _process_and_store_dataset(df, file.filename)


@app.post("/api/sample/{sample_id}", response_model=UploadResponse)
async def load_sample_dataset(sample_id: str):
    """
    Load a pre-packaged multi-domain sample dataset with one click.

    Available domains:
      - retail: E-commerce sales & profit
      - healthcare: Hospital admissions, ICD-10 diagnoses & costs
      - logistics: Freight shipments, carriers, transit & delivery
    """
    sample_files = {
        "retail": "sample_data.csv",
        "healthcare": "sample_healthcare.csv",
        "logistics": "sample_logistics.csv",
    }
    filename = sample_files.get(sample_id.lower())
    if not filename:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sample '{sample_id}'. Available: {list(sample_files.keys())}",
        )

    try:
        df = pd.read_csv(filename)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load sample file '{filename}': {exc}",
        )

    return _process_and_store_dataset(df, filename)


@app.get("/api/baseline-eda")
async def get_baseline_eda():
    """
    Retrieve the automated baseline EDA results (Page 1).

    Returns univariate, bivariate, and multivariate chart payloads.
    """
    if _session["baseline_eda"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV first.",
        )

    baseline: BaselineEDAResult = _session["baseline_eda"]

    def _serialize_payloads(payloads: List[ChartPayload]) -> List[Dict[str, Any]]:
        return [p.model_dump(mode="json") if hasattr(p, "model_dump") else p.dict() for p in payloads]

    return {
        "schema_info": baseline.schema_info,
        "row_count": baseline.row_count,
        "column_count": baseline.column_count,
        "univariate": _serialize_payloads(baseline.univariate),
        "bivariate": _serialize_payloads(baseline.bivariate),
        "multivariate": _serialize_payloads(baseline.multivariate),
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_dataset(request: QueryRequest):
    """
    Page 2: Interactive Natural Language QA.

    Pipeline:
      1. NLP Parser converts question → Structured JSON Query Plan.
      2. Whitelisted Pandas Engine executes the plan.
      3. Visual Selector picks the company-standard chart type.
      4. Summary Engine generates a grounded 1-2 line explanation.
    """
    if _session["df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV first.",
        )

    df: pd.DataFrame = _session["df"]
    parser: NLQueryParser = _session["parser"]
    schema_ctx: SchemaContext = _session["schema_context"]

    # Step 1: Parse NL → JSON Query Plan
    try:
        query_plan: QueryPlan = parser.parse(request.question, schema_ctx)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse the question: {exc}",
        )

    # Step 2: Execute via whitelisted Pandas engine
    try:
        exec_result: ExecutionResult = execute_query_plan(df, query_plan)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Execution error: {exc}",
        )

    # Step 3: Select company-standard visual
    from app.core.field_mapper import ColumnMeta as FieldColMeta, ColumnRole, DtypeClass, map_fields
    from app.core.page1_baseline_eda import classify_columns

    columns_meta = classify_columns(df)

    # Filter metadata to columns actually involved in the query result or plan
    involved_col_names = set(exec_result.columns_used)
    if query_plan.group_by:
        involved_col_names.update(query_plan.group_by)
    if query_plan.agg:
        involved_col_names.add(query_plan.agg.col)
    for f in query_plan.filter:
        involved_col_names.add(f.col)

    query_columns_meta = [c for c in columns_meta if c.name in involved_col_names] or columns_meta

    intent = infer_intent(query_plan, query_columns_meta)
    chart_type = select_visual(
        intent=intent,
        columns=query_columns_meta,
        has_date=any(c.dtype_class == "DATE" for c in query_columns_meta),
        has_geo=any(c.dtype_class == "GEO" for c in query_columns_meta),
        num_numeric=sum(1 for c in query_columns_meta if c.dtype_class == "NUMERIC"),
        num_categorical=sum(1 for c in query_columns_meta if c.dtype_class == "TEXT"),
    )

    # Map fields to visual encoding channels
    try:
        field_meta = [
            FieldColMeta(
                name=c.name,
                dtype_class=DtypeClass(c.dtype_class),
                role=ColumnRole.MEASURE if c.dtype_class == "NUMERIC" else ColumnRole.DIMENSION,
            )
            for c in query_columns_meta
        ]
        field_placement = map_fields(field_meta, chart_type)
    except Exception:
        field_placement = FieldPlacement()

    # Step 4: Generate grounded summary
    # generate_summary expects pd.DataFrame or scalar; executor returns List[Dict].
    # Convert to DataFrame so the summary engine can apply proper grouping logic.
    _result_df: pd.DataFrame = pd.DataFrame(exec_result.result_data) if exec_result.result_data else pd.DataFrame()
    summary = generate_summary(
        result_data=_result_df,
        query_plan=query_plan,
        chart_type=chart_type,
        columns=exec_result.columns_used,
    )

    # Build chart payload
    chart_payload = ChartPayload(
        chart_type=chart_type,
        title=request.question,
        field_placement=field_placement,
        data=exec_result.result_data,
        summary_text=summary,
    )

    return QueryResponse(
        query_plan=query_plan.model_dump(mode="json") if hasattr(query_plan, "model_dump") else query_plan.dict(),
        chart_payload=chart_payload.model_dump(mode="json") if hasattr(chart_payload, "model_dump") else chart_payload.dict(),
        summary=summary,
        intent=intent,
    )


@app.get("/api/schema")
async def get_schema():
    """Return the current dataset schema context."""
    if _session["schema_context"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded.",
        )
    ctx: SchemaContext = _session["schema_context"]
    return {
        "columns": [
            {
                "name": name,
                "dtype_class": ctx.column_types.get(name, "unknown"),
                "unique_count": ctx.cardinalities.get(name, 0),
                "null_count": round(ctx.null_fractions.get(name, 0) * ctx.row_count),
                "sample_values": ctx.sample_values.get(name, [])[:5],
            }
            for name in ctx.column_names
        ],
        "row_count": ctx.row_count,
        "column_count": len(ctx.column_names),
    }


# ---------------------------------------------------------------------------
# Static Files (Frontend)
# ---------------------------------------------------------------------------

try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except Exception:
    pass  # Static directory may not exist yet during tests


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

