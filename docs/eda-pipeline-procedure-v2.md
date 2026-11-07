# Weight-Space Synthesized Natural-Language EDA Engine: Technical Standard Operating Procedure

## 1. Executive Summary & Core Architectural Goal

This Standard Operating Procedure (SOP) defines the technical architecture, data flow, and implementation blueprint for a domain-agnostic **Exploratory Data Analysis (EDA) and Interactive Question-Answering Engine**. The tool is designed to ingest a single, cleaned CSV dataset from any arbitrary domain (e.g., Medical, Transport, Logistics, Finance, Retail) and expose a clean, two-page frontend user interface:

1. **Page 1: Automated Baseline EDA (Univariate, Bivariate & Multivariate Analysis)**
   * Immediately upon CSV upload, the system automatically runs an in-memory statistical and visual profiling suite. 
   * It presents a comprehensive, multi-tiered default analysis consisting of **Univariate Analysis** (single-variable distributions & KPI metric cards), **Bivariate Analysis** (two-variable relationships, time-series trends, categorical comparisons, correlation scatter plots), and **Multivariate Analysis** (multi-dimensional bubble charts, hierarchical matrix/pivot tables, faceted breakdowns).
   * This default page gives users an instant, complete understanding of dataset structure, distributions, and baseline trends without requiring manual configuration.

2. **Page 2: Interactive Natural Language QA & Custom Exploration**
   * Allows users to ask specific natural language questions about the dataset, narrowing down data slices to their exact analytical requirements.
   * Leverages a small domain model pre-steered via **Weight-Space Programming** to translate queries into secure **Structured JSON Query Plans**.
   * Executes computations via a whitelisted Pandas engine (mapped to Power BI DAX formulas), selects company-standard visual chart types, applies strict field placement memory rules, and generates a grounded 1-to-2 line natural language result summary.

To eliminate extensive gradient-descent fine-tuning (SGD) overhead, GPU compute costs, and catastrophic forgetting, the core domain model is initialized and steered using **Weight-Space Programming** (*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, Takehi et al., MIT CSAIL, 2026). Under this paradigm, a frontier LLM directly inspects and edits neural network parameter tensors in one-shot or with minimal forward passes, producing a specialized pre-steered model tailored to the target dataset and query domain.

---

## 2. Integrated Source Reference Matrix

The technical design directly operationalizes all five reference sources provided in the company standard repository:

| Reference Source Document | Operational Role in Procedure | Extracted Standards & Mechanics |
| :--- | :--- | :--- |
| **`natural-language-csv-qa.md`** | System Execution Architecture & Security | Defines the **Structured JSON Query Plan** schema and **Whitelisted Pandas Execution Engine**. Prevents arbitrary code execution (`eval()` / `exec()`) vulnerabilities by separating intent planning from math computation. |
| **`DAX Expressions.pdf`** | Analytical Computation Logic | Provides exact DAX mathematical, aggregation, logical, and time intelligence formulas (`SUM`, `AVERAGE`, `COUNTROWS`, `DISTINCTCOUNT`, `CALCULATE`, `TOTALYTD`, `STDEV.P`, `VAR.P`) translated into deterministic Pandas functions. |
| **`Choose Right Visual Guide.pdf`** | Company Visual Selection Standards | Maps query analytical intent (e.g., categorical comparison, trend over time, proportion breakdown, correlation, distribution) to the exact company-approved visual chart type across Page 1 and Page 2. |
| **`PowerBI_Visual_Field_Placement_CheatSheet_Final.pdf`** | Structural Visual Field Mapping | Dictates exact memory rules for assigning dataset fields to UI slots (e.g., `Text` $\rightarrow$ Axis/Legend, `Number` $\rightarrow$ Values, `Date` $\rightarrow$ Time Axis, `Geo` $\rightarrow$ Map Location). |
| **`Sparks of Weight-Space Programming`** | Model Parameter Steering & Pretraining | Replaces gradient training loops with direct Weight-Space Programming. Enables frontier LLMs to read, write, and edit neural network weights (`W1`, `W2`, `b1`) directly based on natural language specifications and schema contexts. |

---

## 3. Weight-Space Model Synthesis & Direct Initialization (`Sparks of Weight-Space Programming`)

Traditional model adaptation requires collecting domain-specific datasets, defining loss functions, and running multi-epoch gradient descent (SGD/Adam). For this project, we implement **Weight-Space Programming** to directly synthesize and steer the small domain model.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          WEIGHT-SPACE PROGRAMMING LAYER                           │
│                                                                                   │
│  Dataset Schema + Target Task Spec  ──►  Frontier LLM (Weight-Space Programmer)   │
│                                                   │                               │
│                                                   ▼ Direct Tensor Generation      │
│                                       [Weight Tensor Payload: W1, W2, b1]         │
│                                                   │                               │
│                                                   ▼ Direct Parameter Injection    │
│                                     Pre-steered Compact Domain Model              │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Synthesis Mechanics & Parameter Allocation
1. **Direct Parameter Writing**: As established by Takehi et al. (2026), frontier LLMs can natively write functional neural network weights directly from text specifications without gradient descent. The LLM acts as a hyper-optimizer, writing weight matrices (`W1`, `W2`, `b1`) for the small domain model to align its attention/MLP layers with the specific CSV schema.
2. **Zero-Data Capability Steering**: The Weight-Space Programmer steers the small model to infer domain context zero-shot (e.g., recognizing medical ICD diagnostic codes vs. transport route IDs vs. financial ledger metrics).
3. **Targeted Weight Edits**: To adapt or repair reasoning behavior (such as preventing repetitive reasoning loops or fixing parsing edge cases), the LLM applies targeted edits to a tiny fraction (e.g., 0.016% to 3.5%) of the weight tensors rather than retraining the whole network.
4. **Execution Safety**: The Weight-Space Programmer runs inside a sandboxed Python kernel operating over weight tensors, injecting synthesized weights directly into the compact inference model.

---

## 4. System Navigation & Frontend Architecture

The application interface is divided into two distinct pages to decouple automated baseline discovery from custom interactive queries:

```
                                  ┌───────────────────────────┐
                                  │   Single Cleaned CSV      │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ In-Memory Schema Profiler │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
┌──────────────────────────────────────────────┐             ┌──────────────────────────────────────────────┐
│  PAGE 1: AUTOMATED BASELINE EDA              │             │  PAGE 2: INTERACTIVE CUSTOM EXPLORATION      │
├──────────────────────────────────────────────┤             ├──────────────────────────────────────────────┤
│ 1. Univariate Analysis                       │             │ 1. Natural Language Query Input              │
│    - KPI Cards (Totals, Averages, Counts)    │             │ 2. Weight-Space Steered Model Parsing        │
│    - Histograms & Bar Frequency Plots        │             │    - Generates Structured JSON Query Plan    │
│                                              │             │ 3. Whitelisted Pandas Execution (DAX Rules)  │
│ 2. Bivariate Analysis                        │             │ 4. Company-Standard Visual & Slot Selection │
│    - Time-Series Trends (Date vs Metric)     │             │    - Chart Type + Field Placement Memory    │
│    - Categorical Comparisons (Grouped Bar)   │             │ 5. Grounded Summary Delivery                 │
│    - Correlation Scatter Plots (Num vs Num) │             │    - Visual Output + 1-2 Line Explanation    │
│                                              │             │                                              │
│ 3. Multivariate Analysis                     │             │                                              │
│    - Multi-Axis Bubble Charts (X,Y,Size,Col) │             │                                              │
│    - Hierarchical Matrix / Pivot Tables      │             │                                              │
│    - Faceted Grouped & Stacked Breakdowns    │             │                                              │
└──────────────────────────────────────────────┘             └──────────────────────────────────────────────┘
```

---

## 5. Page 1 Procedure: Automated Baseline EDA (Univariate, Bivariate & Multivariate)

When a cleaned CSV dataset is uploaded, Page 1 automatically executes in-memory statistical and visual profiling to generate three analytical tiers.

### 5.1 Metadata & Schema Extraction
* **In-Memory Loading**: The CSV is ingested into an in-memory DataFrame (`natural-language-csv-qa.md`). No persistent dataset storage is required.
* **Column Data Type Classification**:
  * **Categorical / Text**: Discrete strings, codes, category labels (`Text` memory rule).
  * **Quantitative / Numeric**: Continuous measures, amounts, counts (`Number` memory rule).
  * **Chronological / Date**: Timestamps, ISO dates, years, months (`Date` memory rule).
  * **Spatial / Geographic**: Location names, zip codes, coordinates (`Geo` memory rule).

### 5.2 Analytical Tier 1: Univariate Analysis
* **Objective**: Profile individual columns independently to establish metrics, ranges, and distributions.
* **DAX Mathematical Mapping (`DAX Expressions.pdf`)**:
  * Row Count: `COUNTROWS(Table)` $\rightarrow$ `len(df)`
  * Total Sum: `SUM(Column)` $\rightarrow$ `df[col].sum()`
  * Mean Average: `AVERAGE(Column)` $\rightarrow$ `df[col].mean()`
  * Distinct Count: `DISTINCTCOUNT(Column)` $\rightarrow$ `df[col].nunique()`
  * Range Limits: `MIN(Column)`, `MAX(Column)` $\rightarrow$ `df[col].min()`, `df[col].max()`
  * Standard Variance: `STDEV.P(Column)` $\rightarrow$ `df[col].std(ddof=0)`
* **Visual Formatting & Field Placement (`Choose Right Visual Guide`, `PowerBI CheatSheet`)**:
  * **KPI Cards**: Single numeric metrics displayed in high-impact Card visuals (`Numeric` $\rightarrow$ `Values`).
  * **Histograms / Bar Frequency Charts**: Continuous numeric or categorical distributions (`Category/Bin` $\rightarrow$ `Axis/X-Axis`, `Frequency Count` $\rightarrow$ `Values/Y-Axis`).

### 5.3 Analytical Tier 2: Bivariate Analysis
* **Objective**: Evaluate pairwise relationships between two variables to detect trends, comparisons, and correlations.
* **Visual Scenarios & Field Placement**:
  * **Time-Series Trend (Date vs. Metric)**: `Line Chart` or `Column Chart`. (`Date` $\rightarrow$ `Time Axis/X-Axis`, `Numeric Metric` $\rightarrow$ `Y-Axis/Values`). Mapped to DAX time intelligence concepts (`TOTALYTD`, `PREVIOUSMONTH`).
  * **Categorical Comparison (Category vs. Metric)**: `Bar Chart` or `Clustered Column Chart`. (`Text Category` $\rightarrow$ `Axis`, `Numeric Metric` $\rightarrow$ `Values`).
  * **Correlation & Cluster Analysis (Metric vs. Metric)**: `Scatter Plot`. (`Numeric X` $\rightarrow$ `X-Axis`, `Numeric Y` $\rightarrow$ `Y-Axis`, `Category` $\rightarrow$ `Details/Color`).

### 5.4 Analytical Tier 3: Multivariate Analysis
* **Objective**: Explore multi-variable interactions across three or more fields simultaneously.
* **Visual Scenarios & Field Placement**:
  * **Multi-Axis Bubble Chart**: Visualizes 4 dimensions simultaneously (`Numeric X` $\rightarrow$ `X-Axis`, `Numeric Y` $\rightarrow$ `Y-Axis`, `Numeric Size Metric` $\rightarrow$ `Size`, `Category Text` $\rightarrow$ `Color/Legend`).
  * **Hierarchical Matrix & Pivot Table**: Multi-level grouped aggregation view (`Category 1 & Category 2` $\rightarrow$ `Rows/Hierarchy`, `Numeric Measures` $\rightarrow$ `Columns/Values`).
  * **Faceted / Stacked Breakdown**: Segmented performance over time (`Date` $\rightarrow$ `X-Axis`, `Numeric Metric` $\rightarrow$ `Y-Axis`, `Segment Text` $\rightarrow$ `Legend/Stack`).

---

## 6. Page 2 Procedure: Interactive Natural Language QA & Custom Exploration

Page 2 provides an interactive conversational workspace where users ask targeted questions to narrow down data slices.

### 6.1 Step 1: Query Input & Intent Parsing into Structured JSON Plan
To eliminate arithmetic hallucinations and raw code execution security risks (`exec()`), the pre-steered small model converts the user prompt into a strict **Structured JSON Query Plan** (`natural-language-csv-qa.md`):

```json
{
  "operation": "aggregation",
  "filter": [
    {"col": "Status", "op": "==", "value": "Completed"}
  ],
  "group_by": ["MonthName"],
  "agg": {"col": "Revenue", "fn": "sum"},
  "sort": {"by": "Revenue", "desc": true},
  "limit": 10,
  "time_intelligence": "TOTALYTD"
}
```

### 6.2 Step 2: Whitelisted Deterministic Execution Engine
The backend executes the JSON plan against the Pandas DataFrame using a strictly whitelisted function map matching DAX formulas:
* **Operators**: `==`, `!=`, `>`, `>=`, `<`, `<=`
* **Aggregations**: `sum`, `mean`, `count`, `distinct_count`, `min`, `max`, `std`
* **DAX Filter Context Mapping (`CALCULATE`)**:
  * `CALCULATE(SUM(Sales), Region == 'West')` $\rightarrow$ `df[df['Region'] == 'West']['Sales'].sum()`
* **Time Intelligence (`TOTALYTD`, `PREVIOUSMONTH`)**:
  * Joins auto-generated Date Dimension tables to compute year-to-date and period-over-period metrics.

### 6.3 Step 3: Company-Standard Visual Selection & Field Placement Engine
The system maps the query intent to company-approved chart types and field slots:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          VISUAL SELECTION & FIELD PLACEMENT                       │
│                                                                                   │
│   Query Intent / Data Shape                 Target Visual         Field Placement │
│  ───────────────────────────────────────────────────────────────────────────────  │
│   Category vs Metric Comparison         ──► Bar Graph         ──► Y: Text, X: Num │
│   Metric Trend Over Time                ──► Column / Line     ──► X: Date, Y: Num │
│   Cumulative Metric Growth              ──► Area Chart        ──► X: Date, Y: Num │
│   Single Important Figure               ──► Card Visual       ──► Value: Numeric  │
│   Multiple Metrics Comparison           ──► Multi-Row Card    ──► Values: Num/Text│
│   Proportion / Share Analysis           ──► Pie / Donut Chart ──► Legend: Text,   │
│                                                                   Values: Numeric │
│   Two Numeric Variables (Correlation)   ──► Scatter Chart     ──► X: Num, Y: Num  │
│   Three / Four Numeric Variables        ──► Bubble Chart      ──► X,Y,Size,Color  │
│   Single Variable Distribution          ──► Histogram         ──► X: Num, Y: Freq │
│   Multi-Level Hierarchical Breakdown    ──► Matrix / Treemap  ──► Rows/Group: Text│
│   Sequential Incremental Impact         ──► Waterfall Chart   ──► Cat: Text, Y:Num│
│   Target Goal Tracking                  ──► Gauge / KPI       ──► Val, Target: Num│
│   Spatial / Geographic Plotting         ──► Map / Filled Map  ──► Loc: Geo Text   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### Field Placement Memory Rules:
1. **Text / Categorical** $\rightarrow$ `Axis` / `Legend` / `Group`
2. **Numeric Quantitative** $\rightarrow$ `Values` / `Y-Axis` / `Size`
3. **Date / Time** $\rightarrow$ `Time Axis` / `X-Axis`
4. **Geographic** $\rightarrow$ `Location` / `Color`
5. **Secondary Information** $\rightarrow$ `Tooltip`

### 6.4 Step 4: Grounded Result Summary Delivery
A lightweight summary module appends a 1-to-2 line natural language explanation directly referencing the computed numeric output:
* *Example Output*: *"Total Revenue for Completed orders reached $1.42M in October, representing a 14.2% increase compared to September ($1.24M), driven primarily by the Western region."*

---

## 7. Project Directory Layout & Implementation Blueprint

```
eda_cloud_project/
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── core/
│   │   ├── weight_programmer.py    # Weight-Space Programming & parameter steering
│   │   ├── page1_baseline_eda.py   # Page 1 Univariate/Bivariate/Multivariate generator
│   │   ├── nlp_parser.py           # Page 2 Natural Language -> JSON Query Plan parser
│   │   ├── executor.py             # Whitelisted Pandas / DAX execution engine
│   │   ├── visual_selector.py      # Company-standard visual decision engine
│   │   ├── field_mapper.py         # Slot placement rules (Text->Axis, Num->Value)
│   │   └── summary_engine.py       # Grounded 1-2 line result summarizer
│   └── schemas/
│       ├── json_plan_schema.py     # Pydantic schema for structured query plan
│       └── visual_payload.py       # Chart payload specifications (Plotly/Chart.js)
├── tests/
│   ├── test_baseline_eda.py        # Validation for Univariate/Bivariate/Multivariate engine
│   ├── test_dax_pandas.py          # Unit tests for DAX-to-Pandas translation
│   └── test_visual_rules.py        # Validation tests for visual selection
└── procedure_spec.md               # This Standard Operating Procedure document
```

---

## 8. Developer Verification Checklist

Before deploying the pipeline to cloud infrastructure (FastAPI on Google Cloud Run):

- [ ] **Weight-Space Initialization**: Confirm small model parameter tensors (`W1`, `W2`, `b1`) can be injected zero-shot via the Weight-Space Programmer script.
- [ ] **Page 1 Automated Baseline EDA**: Verify CSV upload automatically triggers Univariate (KPI Cards, Histograms), Bivariate (Time-Series, Categorical Bar, Scatter), and Multivariate (Bubble, Matrix, Stacked) chart generation.
- [ ] **Page 2 Interactive QA Flow**: Verify user questions parse into strict JSON plans and execute without raw code injection risks (`eval()`, `exec()`).
- [ ] **DAX Math Accuracy**: Validate Pandas implementations of `SUM`, `AVERAGE`, `COUNTROWS`, `DISTINCTCOUNT`, `STDEV.P`, `VAR.P`, and `TOTALYTD` match Power BI baseline logic.
- [ ] **Company Visual Standards**: Ensure chart selection matches `Choose Right Visual Guide.pdf` and field slot placement adheres to `PowerBI_Visual_Field_Placement_CheatSheet_Final.pdf`.
- [ ] **Summary Grounding**: Confirm 1-to-2 line summaries strictly reference calculated execution numbers and contain zero ungrounded statements.
