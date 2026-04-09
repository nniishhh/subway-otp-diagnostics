# Subway OTP Diagnostics

A Streamlit app for diagnosing subway on-time performance issues with visual diagnostics and Vertex AI analysis.

The app runs a structured Python pre-analysis first, then sends the summarized evidence to Vertex AI for interpretation.

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

## Main Files

- `app.py` — Streamlit UI
- `pre_analysis.py` — Python pre-analysis and chart-ready summaries
- `gemini_helper.py` — Vertex AI request logic
- `prompts.py` — prompt text and JSON-safe helpers
- `requirements.txt` — dependencies

## Deploy To Cloud Run

This repo includes Docker and GitHub Actions setup so every push to `main` can redeploy the app to Cloud Run.

### Google Cloud Setup

Use project `ieor-oncloud`.

1. Enable these APIs:
   - Cloud Run Admin API
   - Artifact Registry API
   - Cloud Build API
   - Vertex AI API
   - IAM Service Account Credentials API

2. Create an Artifact Registry repository in `us-east1`:
   - name: `subway-otp-diagnostics`

3. Create one service account for both deploy and runtime.

4. Give that service account permissions for:
   - Cloud Run deploy
   - Artifact Registry push
   - Vertex AI runtime access
   - Service Account User

5. Download the JSON key for that service account.

### GitHub Secrets

Set these in:
`GitHub -> Settings -> Secrets and variables -> Actions`

- `GCP_PROJECT_ID=ieor-oncloud`
- `GCP_REGION=us-east1`
- `GCP_SERVICE_NAME=subway-otp-diagnostics`
- `GCP_SA_KEY=<full service account JSON>`

Artifact Registry repo name is fixed in the workflow as:

- `subway-otp-diagnostics`

### Auto-Deploy Behavior

- pushes to `main` trigger `.github/workflows/deploy.yml`
- the workflow builds a container image
- the image is pushed to Artifact Registry
- the app is deployed to Cloud Run with a public URL

### Cloud Run Runtime Config

The deploy workflow sets:

- `VERTEXAI_PROJECT=ieor-oncloud`
- `VERTEXAI_LOCATION=global`
- `VERTEXAI_MODEL=gemini-3-flash-preview`

Cloud Run should use the attached service account for ADC at runtime. Do not set `GOOGLE_APPLICATION_CREDENTIALS` in Cloud Run.

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

## Notes

- the app is sample-first and currently uses `sample_data/`
- `data/` was removed so there is one source of truth for scenarios
- if Vertex AI fails, the app shows the failure instead of inventing a fallback diagnosis
- local `.env` is for development only and is not used by the Cloud Run deployment
