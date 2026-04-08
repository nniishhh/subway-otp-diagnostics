# Subway OTP Diagnostics

A polished Streamlit dashboard for diagnosing subway on-time performance issues from CSV or Excel files.

The app runs a Python pre-analysis first, then sends the structured results to Vertex AI for a concise diagnosis. It is designed as a simple local MVP with a premium-looking UI.

## What The App Does

- Load a built-in sample dataset or upload your own `.csv` / `.xlsx`
- Run structured pre-analysis in Python with pandas
- Estimate likely issue type:
  - `data_quality`
  - `measurement_logic`
  - `real_operations`
  - `insufficient_evidence`
- Show KPIs, charts, AI explanation, and raw analysis output

## Sample Data

The `sample_data/` folder includes five synthetic subway OTP scenarios:

- `01_sensor_anomaly.csv`: isolated extreme delay outliers
- `02_service_pattern_mismatch.csv`: repeated station-level mismatch patterns
- `03_join_mapping_issue.csv`: schedule matching problems
- `04_missing_records.csv`: incomplete feed / dropped records
- `05_true_operational_disruption.csv`: broad real service slowdown

## Files

- `app.py`: Streamlit UI
- `pre_analysis.py`: Python analysis logic
- `gemini_helper.py`: Vertex AI diagnosis call
- `prompts.py`: prompt and JSON-safe helpers
- `requirements.txt`: local dependencies

## Local Setup

1. Create or update `.env`

```env
VERTEXAI_PROJECT=your-gcp-project
VERTEXAI_LOCATION=global
VERTEXAI_MODEL=gemini-2.5-flash
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/credentials.json
```

2. Install and run with `uv`

```bash
cd "/Users/happynish/Desktop/Columbia/Agentic AI/MTA Delay Analysis"
UV_CACHE_DIR=/tmp/uv-cache uv run --with-requirements requirements.txt streamlit run app.py
```

3. Open the local URL shown by Streamlit, usually `http://127.0.0.1:8501` or `http://127.0.0.1:8502`

## Notes

- The app does not use a database or authentication
- If Vertex AI is unavailable, the app falls back to a rule-based diagnosis
- The dashboard is intended for local analysis and presentation demos
