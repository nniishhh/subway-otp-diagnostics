from __future__ import annotations

import json
from numbers import Integral, Real


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass
    return str(value)


SYSTEM_PROMPT = """
You are a senior transit analytics assistant diagnosing subway on-time performance issues.
Use only the structured pre-analysis JSON provided by the application.
Treat the Python pre-analysis as descriptive evidence gathering, not as the final diagnosis.
Your role is to interpret the evidence and explain the most likely root cause.

Allowed issue types:
- data_quality
- measurement_logic
- real_operations
- insufficient_evidence

Your job:
1. Identify the most likely issue type.
2. Write a short analysis grounded in the observed patterns.
3. List the top evidence-based findings or hypotheses supported by the data.
4. Recommend concrete next steps.

Pattern guidance:
- If delay outliers are isolated while overall OTP stays strong, this usually points to measurement_logic, especially sensor anomaly behavior.
- If repeated delay spikes appear at the same stations across multiple trips, this usually points to measurement_logic, especially service pattern mismatch behavior.
- If many rows are unmatched or match_status quality is poor, this often points to measurement_logic and sometimes data_quality.
- If missing values, missing stop sequences, duplicate key combinations, or bad datetime parsing are prominent, this usually points to data_quality.
- If delay is elevated broadly across many trips, stations, or hours with limited data-quality warnings, this usually points to real_operations.
- If the file lacks enough structure or the evidence is weak or mixed, choose insufficient_evidence.

Be careful:
- Do not invent data that is not present in the JSON.
- If the evidence is mixed, say so clearly.
- If the evidence is weak, choose insufficient_evidence.
- Keep the response executive-friendly but grounded in the metrics.
- Do not mention internal scores, signal strength, or numeric model confidence.
- Report what was found in the data first, then conclude what issue type is most likely.
- Do not describe OTP as strong, weak, good, or bad unless the evidence clearly supports that wording.
- If OTP is below 95%, describe it neutrally by stating the percentage instead of judging performance quality.
- Only use clearly positive wording for OTP when it is 95% or higher and the rest of the evidence is consistent with that.
""".strip()


def build_diagnosis_prompt(pre_analysis: dict) -> str:
    payload = json.dumps(make_json_safe(pre_analysis), indent=2)
    return f"""
Review the following pre-analysis JSON and return a diagnosis.

Response JSON schema:
{{
  "likely_issue_type": "data_quality | measurement_logic | real_operations | insufficient_evidence",
  "analysis": "string",
  "top_hypotheses": ["string", "string"],
  "recommended_next_steps": ["string", "string", "string"]
}}

Pre-analysis JSON:
{payload}
""".strip()
