# Mini-EDA-Model-Cloud-Computing
### Weight-Space Synthesized Natural-Language EDA Engine

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-55%20Passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Cloud%20Run%20Ready-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A domain-agnostic **Exploratory Data Analysis (EDA) and Interactive Natural-Language Question-Answering Engine**, powered by **Weight-Space Programming** (*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*, Takehi et al., MIT CSAIL 2026).

---

## 🚀 Core Features

- **⚡ Weight-Space Programming Layer**: Replaces gradient-descent fine-tuning (SGD) with one-shot direct weight tensor synthesis ($W_1, W_2, b_1$), aligning the domain model to target CSV schemas without catastrophic forgetting.
- **📊 Page 1: Automated Baseline EDA**:
  - **Univariate Analysis**: Single-variable distributions, KPI cards (`SUM`, `AVERAGE`, `COUNTROWS`, `MIN`, `MAX`, `STDEV.P`), and frequency plots.
  - **Bivariate Analysis**: Two-variable relationships, time-series trends (`TOTALYTD`, `PREVIOUSMONTH`), categorical bars, and correlation scatter plots.
  - **Multivariate Analysis**: 4-variable bubble charts, hierarchical matrices/pivot tables, and faceted breakdowns.
- **💬 Page 2: Interactive Natural Language QA**:
  - Pre-steered NLP parser translates English questions into secure **Structured JSON Query Plans**.
  - **Whitelisted Pandas/DAX Execution Engine**: Strictly zero `eval()` or `exec()` code injection risks.
  - **Company-Standard Visual Selection**: Implements the *Choose Right Visual Guide* (Category $\rightarrow$ Bar, Trend $\rightarrow$ Line, Proportion $\rightarrow$ Donut/Pie, Single Value $\rightarrow$ Card, Geographic $\rightarrow$ Map).
  - **Field Placement Memory Rules**: Categorical $\rightarrow$ Axis/Legend, Numeric $\rightarrow$ Values/Y-Axis, Date $\rightarrow$ Time Axis, Geographic $\rightarrow$ Location.
  - **Grounded 1–2 Line Summary Engine**: Factually explains outcomes directly from computed numbers with currency and ratio formatting.
- **🎨 Interactive Web Interface**: Dark mode, responsive frontend built with Plotly.js for interactive chart exploration.
- **☁️ Cloud Run Production Ready**: Multi-stage Docker containerization with port parameterization.

---

## 🏛️ System Architecture

```
                                      CSV Upload
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │      Schema Profiler & WSP      │
                         │    (weight_programmer.py)       │
                         └──────────────┬──────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │ Synthesised Domain Weights                 │ Profiler Metadata
                 ▼                                             ▼
  ┌─────────────────────────────┐               ┌─────────────────────────────┐
  │  PAGE 2: INTERACTIVE QA     │               │  PAGE 1: AUTOMATED BASELINE │
  │                             │               │                             │
  │  • NLP Query Parser         │               │  • Univariate Analysis      │
  │    (pre-steered by WSP)     │               │    (KPI Cards, Histograms)  │
  │  • Whitelisted DAX Executor │               │  • Bivariate Analysis       │
  │    (zero eval / exec)       │               │    (Time Series, Bar, Scat) │
  │  • Visual Selector (SOP)    │               │  • Multivariate Analysis    │
  │  • Slot Placement Mapper    │               │    (Bubble, Matrix, Heatmap)│
  │  • Grounded Summary Engine  │               └─────────────────────────────┘
  └─────────────────────────────┘
```

---

## 📂 Project Directory Structure

```
├── app/
│   ├── main.py                     # FastAPI application entrypoint (7 routes)
│   ├── core/
│   │   ├── weight_programmer.py    # Weight-Space Programming (Takehi et al., 2026)
│   │   ├── page1_baseline_eda.py   # Page 1 Univariate/Bivariate/Multivariate generator
│   │   ├── nlp_parser.py           # Page 2 Natural Language -> JSON Query Plan parser
│   │   ├── executor.py             # Whitelisted Pandas / DAX execution engine
│   │   ├── visual_selector.py      # Company-standard visual decision engine
│   │   ├── field_mapper.py         # Slot placement rules (Text->Axis, Num->Value)
│   │   └── summary_engine.py       # Grounded 1-2 line result summarizer
│   ├── schemas/
│   │   ├── json_plan_schema.py     # Pydantic schema for structured query plan
│   │   └── visual_payload.py       # Visual payload schema (Plotly-ready)
│   └── static/
│       └── index.html              # Dark mode Plotly.js frontend UI
├── tests/
│   ├── test_baseline_eda.py        # Tests for Univariate/Bivariate/Multivariate engine
│   ├── test_dax_pandas.py          # Unit tests for DAX-to-Pandas execution
│   └── test_visual_rules.py        # Tests for visual selection & field placement
├── docs/                           # Blueprint, SOP, and company visual guides
├── Dockerfile                      # Google Cloud Run container specification
├── requirements.txt                # Python package dependencies
├── sample_data.csv                 # E-commerce retail dataset (30 rows, 8 cols)
├── sample_healthcare.csv           # Hospital admissions dataset (25 rows, 8 cols)
└── sample_logistics.csv            # Freight logistics dataset (25 rows, 9 cols)
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/WORTHOX/Mini-EDA-Model-Cloud-Computing.git
cd Mini-EDA-Model-Cloud-Computing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Development Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Testing

Run the full automated test suite (55 tests):

```bash
python -m pytest tests/ -v
```

---

## ☁️ Google Cloud Run Deployment

Build and deploy directly using Google Cloud SDK:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy directly from source
gcloud run deploy mini-eda-engine \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

Or build with Docker:

```bash
docker build -t gcr.io/YOUR_PROJECT_ID/mini-eda-engine:latest .
docker run -p 8080:8080 gcr.io/YOUR_PROJECT_ID/mini-eda-engine:latest
```

---

## 📖 Reference Standards

This implementation operationalizes standards from:
- **`Sparks of Weight-Space Programming`** (Takehi et al., MIT CSAIL 2026)
- **`DAX Expressions`**: Deterministic mappings for `SUM`, `AVERAGE`, `COUNTROWS`, `DISTINCTCOUNT`, `STDEV.P`, `TOTALYTD`.
- **`Choose Right Visual Guide`**: Intent-to-visual selection taxonomy.
- **`PowerBI Visual Field Placement CheatSheet`**: Visual slot memory rules.
- **`natural-language-csv-qa`**: Whitelisted query plan security.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
