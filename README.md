# Subway OTP Diagnostics

A Streamlit app for diagnosing subway on-time performance issues with visual diagnostics and Vertex AI (Gemini 3 Flash).

The app runs a structured Python pre-analysis first, then sends the summarized evidence to Vertex AI (Gemini 3 Flash) for interpretation.

Live app:
https://subway-otp-diagnostics-885256127561.us-east1.run.app/

## What It Does

- analyzes synthetic subway OTP scenarios from `sample_data/`
- shows overview metrics and visual diagnostics
- uses Vertex AI to summarize what the data suggests, what happened, and what to check next

## Sample Scenarios

- `01_isolated_extreme_delays.csv`
  - a few stops spike far above the normal delay range
- `02_recurring_bottleneck.csv`
  - the same stations show repeated delay patterns across trips
- `03_schedule_vs_actual_mismatch.csv`
  - schedule records and actual train data do not align cleanly
- `04_missing_trip_records.csv`
  - some trips lose stop updates and end up with broken sequences
- `05_network_wide_slowdown.csv`
  - delays are elevated across many trips and stations at once

## Run Locally

1. Fill in `.env`

```env
VERTEXAI_PROJECT=your-gcp-project-id
VERTEXAI_LOCATION=global
VERTEXAI_MODEL=gemini-3-flash-preview
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/credentials.json
```

2. Install and run

```bash
cd "/Users/happynish/Desktop/Columbia/Agentic AI/MTA Delay Analysis"
UV_CACHE_DIR=/tmp/uv-cache uv run --with-requirements requirements.txt streamlit run app.py
```

3. Open the local Streamlit URL shown in the terminal
