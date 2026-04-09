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


def prompt_payload(pre_analysis: dict) -> dict:
    payload = dict(pre_analysis)
    file_info = dict(payload.get("file_info", {}))
    file_info.pop("file_name", None)
    payload["file_info"] = file_info
    payload.pop("evidence_summary", None)
    return payload


SYSTEM_PROMPT = """
You are a senior transit analytics assistant diagnosing subway on-time performance issues.
Use only the structured pre-analysis JSON provided by the application.
Treat the Python pre-analysis as descriptive evidence gathering, not as the final diagnosis.
Your role is to interpret the evidence and explain the most likely pattern behind what the data shows.

Your job:
1. Write a short summary statement grounded in the observed patterns.
2. Summarize what happened in the data in a few concrete bullets.
3. Recommend concrete next steps.

There are five recognizable scenario patterns you may notice:
- isolated extreme delays
- recurring bottleneck
- schedule vs actual mismatch
- missing trip records
- network-wide slowdown

Pattern guidance:
- If only a few records show extreme delay spikes while most records remain normal, this may indicate an isolated extreme delay pattern.
- If the same stations repeatedly show elevated delay across multiple trips, this may indicate a recurring bottleneck pattern.
- If schedule records and actual train records do not align cleanly, especially with unmatched rows or suspicious timing gaps, this may indicate a schedule vs actual mismatch pattern.
- If trips have broken stop sequences or missing stops, this may indicate a missing trip records pattern.
- If delay is elevated broadly across many trips, stations, or time windows, this may indicate a network-wide slowdown pattern.
- If the evidence is mixed or weak, stay descriptive and do not force a pattern label.

Be careful:
- Do not invent data that is not present in the JSON.
- If the evidence is mixed, say so clearly.
- If the evidence is weak, choose insufficient_evidence.
- Keep the response executive-friendly but grounded in the metrics.
- Do not mention internal scores, signal strength, or numeric model confidence.
- Report what was found in the data first, then cautiously mention a recognizable pattern only if the evidence supports it.
- Do not describe OTP as strong, weak, good, or bad unless the evidence clearly supports that wording.
- If OTP is below 95%, describe it neutrally by stating the percentage instead of judging performance quality.
- Only use clearly positive wording for OTP when it is 95% or higher and the rest of the evidence is consistent with that.
- Write in plain, intuitive language, as if you are briefing an operations manager.
- Keep the writing concise and easy to scan.
- Prefer concrete evidence such as counts, trips, stations, and visible delay patterns over abstract diagnostic jargon.
- The summary line should start with "What the data suggests:" and use tentative wording such as "most likely", "appears to", or "suggests".
- Do not write as if the pattern or root cause is confirmed with certainty.
- The "what happened" bullets should describe what a person would actually notice in the charts or tables, in short intuitive bullets.
""".strip()


def build_diagnosis_prompt(pre_analysis: dict) -> str:
    payload = json.dumps(make_json_safe(prompt_payload(pre_analysis)), indent=2)
    return f"""
Review the following pre-analysis JSON and return a diagnosis.

Response JSON schema:
{{
  "analysis": "string",
  "what_happened": ["string", "string", "string"],
  "recommended_next_steps": ["string", "string", "string"]
}}

Pre-analysis JSON:
{payload}
""".strip()
