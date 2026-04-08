from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from prompts import SYSTEM_PROMPT, build_diagnosis_prompt, make_json_safe


load_dotenv()


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "likely_issue_type": {
            "type": "string",
            "enum": [
                "data_quality",
                "measurement_logic",
                "real_operations",
                "insufficient_evidence",
            ],
        },
        "analysis": {"type": "string"},
        "top_hypotheses": {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "likely_issue_type",
        "analysis",
        "top_hypotheses",
        "recommended_next_steps",
    ],
}


def fallback_diagnosis(pre_analysis: dict, error_message: str | None = None) -> dict:
    fallback = make_json_safe(dict(pre_analysis["evidence_summary"]))
    fallback["analysis"] = fallback.pop("short_explanation", "")
    fallback.pop("confidence", None)
    fallback["source"] = "rule_based"
    if error_message:
        fallback["warning"] = error_message
    return fallback


def run_vertex_diagnosis(pre_analysis: dict, model: str = "gemini-2.5-flash") -> dict:
    project = os.getenv("VERTEXAI_PROJECT")
    location = os.getenv("VERTEXAI_LOCATION", "us-central1")
    model_name = os.getenv("VERTEXAI_MODEL", model)

    if not project:
        return fallback_diagnosis(pre_analysis, "VERTEXAI_PROJECT is not set.")

    try:
        vertexai.init(project=project, location=location)
        model_client = GenerativeModel(
            model_name=model_name,
            system_instruction=[SYSTEM_PROMPT],
        )
        response = model_client.generate_content(
            build_diagnosis_prompt(pre_analysis),
            generation_config=GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        diagnosis = json.loads(response.text)
        if "analysis" not in diagnosis and "short_explanation" in diagnosis:
            diagnosis["analysis"] = diagnosis.pop("short_explanation")
        diagnosis.pop("confidence", None)

        diagnosis["source"] = "vertex_ai"
        return diagnosis
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        return fallback_diagnosis(pre_analysis, f"Vertex AI call failed: {exc}")
