from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompts import SYSTEM_PROMPT, build_diagnosis_prompt


load_dotenv()


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "likely_issue_type": {
            "type": "STRING",
            "enum": [
                "data_quality",
                "measurement_logic",
                "real_operations",
                "insufficient_evidence",
            ],
        },
        "analysis": {"type": "STRING"},
        "top_hypotheses": {"type": "ARRAY", "items": {"type": "STRING"}},
        "recommended_next_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": [
        "likely_issue_type",
        "analysis",
        "top_hypotheses",
        "recommended_next_steps",
    ],
}


def unavailable_diagnosis(error_message: str) -> dict:
    return {
        "likely_issue_type": "insufficient_evidence",
        "analysis": "",
        "top_hypotheses": [],
        "recommended_next_steps": [],
        "source": "vertex_unavailable",
        "warning": error_message,
    }


def normalize_diagnosis(payload: dict) -> dict:
    diagnosis = dict(payload)
    if "analysis" not in diagnosis and "short_explanation" in diagnosis:
        diagnosis["analysis"] = diagnosis.pop("short_explanation")
    diagnosis.pop("confidence", None)
    diagnosis["source"] = "vertex_ai"
    return diagnosis


def run_vertex_diagnosis(
    pre_analysis: dict,
    model: str = "gemini-3-flash-preview",
    max_attempts: int = 3,
) -> dict:
    project = os.getenv("VERTEXAI_PROJECT")
    location = os.getenv("VERTEXAI_LOCATION", "global")
    model_name = os.getenv("VERTEXAI_MODEL", model)

    if not project:
        return unavailable_diagnosis("VERTEXAI_PROJECT is not set.")

    prompt = build_diagnosis_prompt(pre_analysis)
    last_error = "Vertex AI call failed."

    for attempt in range(1, max_attempts + 1):
        try:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

            client = genai.Client(http_options=types.HttpOptions(api_version="v1"))
            config_kwargs = {
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": RESPONSE_SCHEMA,
            }
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            diagnosis = json.loads(response.text)
            return normalize_diagnosis(diagnosis)
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            last_error = f"Vertex AI call failed on attempt {attempt}/{max_attempts}: {exc}"
            if attempt < max_attempts:
                time.sleep(0.75 * attempt)

    return unavailable_diagnosis(last_error)
