from __future__ import annotations
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gemini_helper import run_vertex_diagnosis
from pre_analysis import load_dataframe, run_pre_analysis
from prompts import build_diagnosis_prompt


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
SAMPLE_DESCRIPTIONS = {
    "01_isolated_extreme_delays.csv": {
        "title": "01 Isolated Extreme Delays",
        "summary": "A small number of stops show unusually high delays (10+ min), while most remain normal.",
        "look_for": "A few sharp spikes, not a system-wide pattern",
    },
    "02_recurring_bottleneck.csv": {
        "title": "02 Recurring Bottleneck",
        "summary": "Certain stations consistently experience higher delays across multiple trips.",
        "look_for": "The same stops showing repeated delay patterns",
    },
    "03_schedule_vs_actual_mismatch.csv": {
        "title": "03 Schedule Vs Actual Mismatch",
        "summary": "Scheduled times and real train data do not align, creating artificial delay gaps.",
        "look_for": "Unmatched records or inconsistent timing gaps",
    },
    "04_missing_trip_records.csv": {
        "title": "04 Missing Trip Records",
        "summary": "Some trips have incomplete stop sequences due to missing updates.",
        "look_for": "Broken trip sequences or fewer stops than expected",
    },
    "05_network_wide_slowdown.csv": {
        "title": "05 Network-Wide Slowdown",
        "summary": "Delays are elevated across many trips and stations, indicating a real system issue.",
        "look_for": "Widespread delays, not isolated spikes",
    },
}

CHART_PALETTE = {
    "core_scale": ["#13233d", "#1d4ed8", "#38bdf8", "#a5f3fc"],
    "core_scale_stops": [
        [0.0, "#13233d"],
        [0.3, "#1d4ed8"],
        [0.68, "#38bdf8"],
        [1.0, "#a5f3fc"],
    ],
    "heatmap_scale_stops": [
        [0.0, "#0f172a"],
        [0.24, "#1e3a8a"],
        [0.52, "#2563eb"],
        [0.8, "#f59e0b"],
        [1.0, "#ef4444"],
    ],
    "alert_scale": ["#7c2d12", "#c2410c", "#f59e0b", "#ef4444"],
    "alert_scale_stops": [
        [0.0, "#7c2d12"],
        [0.35, "#c2410c"],
        [0.7, "#f59e0b"],
        [1.0, "#ef4444"],
    ],
    "distribution_scale_stops": [
        [0.0, "#334155"],
        [0.28, "#64748b"],
        [0.62, "#f59e0b"],
        [1.0, "#dc2626"],
    ],
    "otp_scale": ["#b91c1c", "#d97706", "#65a30d"],
    "comparison": ["#93c5fd", "#22d3ee", "#fbbf24", "#34d399"],
    "threshold": "rgba(245, 158, 11, 0.62)",
    "spike": "#fbbf24",
}

SEVERITY_COLOR_MAP = {
    "minor": "#38bdf8",
    "moderate": "#fbbf24",
    "major": "#f97316",
    "critical": "#ef4444",
}

st.set_page_config(
    page_title="Diagnose Subway OTP Issues With Visualization And AI Analysis",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(65, 105, 225, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(251, 146, 60, 0.12), transparent 24%),
                linear-gradient(180deg, #08111a 0%, #09131d 100%);
            color: #edf2ff;
        }
        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }
        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }
        #MainMenu, footer {
            visibility: hidden;
            height: 0;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4, p, label, div {
            color: #edf2ff;
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 28px;
            padding: 2.2rem 2.3rem;
            background:
                linear-gradient(135deg, rgba(13, 20, 39, 0.95), rgba(17, 24, 39, 0.88)),
                linear-gradient(120deg, rgba(96, 165, 250, 0.15), rgba(129, 140, 248, 0.05));
            box-shadow: 0 30px 70px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.5rem;
        }
        .hero-eyebrow {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(96, 165, 250, 0.16);
            color: #e2e8f0;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 3rem;
            line-height: 1.04;
            margin: 0 0 0.8rem 0;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .hero-copy {
            color: #cbd5e1;
            max-width: 900px;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        .hero-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 0.2rem;
        }
        .hero-link {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.7rem 1rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.58);
            color: #e2e8f0;
            font-size: 0.92rem;
            font-weight: 700;
            text-decoration: none;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }
        .hero-link:hover {
            transform: translateY(-1px);
            border-color: rgba(96, 165, 250, 0.42);
            background: rgba(30, 41, 59, 0.78);
            color: #f8fafc;
        }
        .section-card {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            background: rgba(15, 23, 42, 0.78);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #e2e8f0;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }
        .metric-card {
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(28, 39, 61, 0.9), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(148, 163, 184, 0.14);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
            min-height: 92px;
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.32rem;
            font-weight: 800;
            margin-top: 0.28rem;
            color: #f8fafc;
            line-height: 1.1;
        }
        .metric-value-good { color: #4ade80; }
        .metric-value-warn { color: #facc15; }
        .metric-value-bad  { color: #f87171; }
        .metric-caption {
            color: #94a3b8;
            font-size: 0.74rem;
            margin-top: 0.22rem;
            line-height: 1.4;
        }
        .overview-panel {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 22px;
            padding: 1.05rem 1.1rem;
            background: rgba(15, 23, 42, 0.78);
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.2);
            margin-bottom: 0.9rem;
        }
        .overview-kicker {
            color: #e2e8f0;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.6rem;
        }
        .overview-summary {
            font-size: 1.05rem;
            line-height: 1.65;
            color: #e2e8f0;
            margin-top: 0.15rem;
        }
        .sample-card {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.7);
            padding: 1rem;
            min-height: 165px;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.22);
        }
        .sample-card.selected {
            border-color: rgba(96, 165, 250, 0.65);
            box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.25), 0 18px 36px rgba(0, 0, 0, 0.28);
        }
        .sample-group-title {
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            margin: 0.4rem 0 0.2rem 0;
        }
        .sample-group-copy {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 0.85rem;
            max-width: 900px;
        }
        .sample-title {
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
            line-height: 1.35;
        }
        .sample-copy {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.58;
        }
        .sample-summary {
            margin-bottom: 0.75rem;
        }
        .sample-lookfor {
            color: #e2e8f0;
            font-size: 0.88rem;
            line-height: 1.52;
        }
        .sample-lookfor strong {
            color: #f8fafc;
        }
        .insight-copy {
            color: #dbe4f0;
            font-size: 1rem;
            line-height: 1.6;
        }
        .list-card {
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.12);
            padding: 1rem 1rem 0.7rem 1rem;
            height: 100%;
        }
        .list-card ul {
            margin: 0.3rem 0 0 1.1rem;
        }
        .list-card li {
            color: #dbe4f0;
            margin-bottom: 0.75rem;
            line-height: 1.6;
        }
        .list-card li:last-child {
            margin-bottom: 0.2rem;
        }
        .pill-note {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
            color: #cbd5e1;
            font-size: 0.84rem;
        }
        div.stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(96, 165, 250, 0.22);
            background: linear-gradient(180deg, rgba(37, 99, 235, 0.92), rgba(29, 78, 216, 0.92));
            color: white;
            font-weight: 700;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
        }
        div.stButton > button:hover {
            border-color: rgba(147, 197, 253, 0.5);
            color: white;
        }
        details[data-testid="stExpander"] summary {
            color: #e2e8f0 !important;
        }
        details[data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.9), rgba(10, 15, 28, 0.86));
        }
        .stFileUploader > div {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 18px;
            border: 1px dashed rgba(148, 163, 184, 0.25);
        }
        div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.82);
            border-color: rgba(148, 163, 184, 0.18);
            color: #e2e8f0;
        }
        div[data-baseweb="select"] input {
            color: #e2e8f0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.72);
            border-radius: 16px;
            padding: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            color: #e2e8f0;
            padding: 0.7rem 1rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.2);
            color: #eff6ff;
            border: 1px solid rgba(96, 165, 250, 0.25);
        }
        .stTabs [aria-selected="false"]:hover {
            background: rgba(148, 163, 184, 0.08);
            color: #f1f5f9;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        .input-shell {
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 24px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            background: rgba(15, 23, 42, 0.78);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
            min-height: 100%;
        }
        .active-file-card {
            border: 1px solid rgba(96, 165, 250, 0.18);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.86), rgba(15, 23, 42, 0.84));
            margin: 0.75rem 0 0.9rem 0;
        }
        .active-file-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #93c5fd;
            margin-bottom: 0.32rem;
            font-weight: 700;
        }
        .active-file-name {
            font-size: 1rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .active-file-meta {
            font-size: 0.82rem;
            color: #cbd5e1;
            margin-top: 0.24rem;
        }
        .empty-state {
            text-align: center;
            padding: 2.2rem 1.2rem;
            border: 1px dashed rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.54);
            color: #cbd5e1;
        }
        .prompt-preview {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(10, 15, 28, 0.92));
            padding: 1rem 1.05rem;
            max-height: 420px;
            overflow: auto;
        }
        .prompt-preview pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: #cbd5e1;
            font-size: 0.86rem;
            line-height: 1.55;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Subway OTP Diagnostics</div>
            <h1 class="hero-title">Diagnose subway OTP issues with visualization and AI analysis.</h1>
            <p class="hero-copy">
                This app analyzes subway on-time performance scenarios and helps explain whether the pattern looks like
                isolated extreme delays, a recurring bottleneck, a schedule-vs-actual mismatch, missing trip records,
                or a broader network slowdown. It works by running a Python pre-analysis first, then sending the structured
                findings to Gemini for interpretation.
            </p>
            <div class="hero-links">
                <a class="hero-link" href="https://github.com/nniishhh/subway-otp-diagnostics" target="_blank" rel="noopener noreferrer">View Source Code</a>
                <a class="hero-link" href="https://www.youtube.com/watch?v=kLziIIrv_NA" target="_blank" rel="noopener noreferrer">Watch Demo Video</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str, value_class: str = "") -> str:
    value_cls = f"metric-value {value_class}".strip()
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{value_cls}">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def _otp_color_class(otp: float | None) -> str:
    if otp is None:
        return ""
    if otp >= 85:
        return "metric-value-good"
    if otp >= 70:
        return "metric-value-warn"
    return "metric-value-bad"


def render_metric_grid(pre_analysis: dict, diagnosis: dict) -> None:
    metrics = pre_analysis["performance_metrics"]
    otp_percent = metrics.get("otp_percent")
    avg_delay = metrics.get("avg_delay")
    max_delay = metrics.get("max_delay")
    outlier_count = metrics.get("outlier_count", 0)
    trip_count = metrics.get("trip_count")
    station_count = metrics.get("station_count")
    cards = [
        (
            "Rows",
            f"{pre_analysis['file_info']['row_count']:,}",
            "Records analyzed",
            "",
        ),
        (
            "OTP",
            f"{otp_percent:.1f}%" if otp_percent is not None else "n/a",
            "Using delay \u2264 5 min",
            _otp_color_class(otp_percent),
        ),
        (
            "Avg Delay",
            f"{avg_delay:.2f} min" if avg_delay is not None else "n/a",
            f"Max {max_delay:.1f} min" if max_delay is not None else "Average delay unavailable",
            "",
        ),
        (
            "Outliers",
            f"{outlier_count:,}",
            "IQR-based delay outliers",
            "",
        ),
        (
            "Trips / Stations",
            f"{trip_count or 'n/a'} / {station_count or 'n/a'}",
            "Unique IDs detected",
            "",
        ),
    ]
    columns = st.columns(5, gap="small")
    for index, (label, value, caption, value_class) in enumerate(cards):
        with columns[index]:
            st.markdown(metric_card(label, value, caption, value_class), unsafe_allow_html=True)


def render_sample_selector() -> tuple[Path | None, bool]:
    st.markdown("<div class='section-title'>Choose A Sample Dataset</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pill-note" style="margin-top:-0.15rem;margin-bottom:1rem;">
            Pick one of the five generated scenario files below to run the analysis immediately. These synthetic datasets were created to mimic common subway delay patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )
    sample_files = sorted(SAMPLE_DIR.glob("*"))
    if not sample_files:
        st.info("No sample files found in sample_data.")
        return None, False
    sample_lookup = {sample_file.name: sample_file for sample_file in sample_files}

    selected = st.session_state.get("selected_sample")
    analyze_requested = False
    columns = st.columns(3)
    for index, sample_file in enumerate(sample_files):
        with columns[index % 3]:
            is_selected = selected == sample_file.name
            selected_class = "selected" if is_selected else ""
            description = SAMPLE_DESCRIPTIONS.get(
                sample_file.name,
                {
                    "title": sample_file.stem.replace("_", " ").title(),
                    "summary": "Built-in scenario dataset.",
                    "look_for": "A notable transit delay pattern in the visuals.",
                },
            )
            st.markdown(
                f"""
                <div class="sample-card {selected_class}">
                    <div class="sample-title">{description['title']}</div>
                    <div class="sample-copy">
                        <div class="sample-summary">{description['summary']}</div>
                        <div class="sample-lookfor"><strong>Look for:</strong> {description['look_for']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Run Analysis",
                key=f"sample-{sample_file.name}",
                use_container_width=True,
            ):
                st.session_state["selected_sample"] = sample_file.name
                selected = sample_file.name
                analyze_requested = True

    if selected:
        return SAMPLE_DIR / selected, analyze_requested
    return None, analyze_requested


def get_active_file(sample_path: Path | None) -> tuple[object | None, str | None]:
    if sample_path and sample_path.exists():
        return sample_path, sample_path.name
    return None, None


def run_analysis(file_source: object, file_name: str) -> None:
    with st.spinner("Running pre-analysis and preparing the Gemini prompt..."):
        frame, metadata = load_dataframe(file_source, file_name)
        pre_analysis = run_pre_analysis(frame, metadata["file_name"])
        prompt_input = build_diagnosis_prompt(pre_analysis)
        diagnosis = run_vertex_diagnosis(pre_analysis)

    st.session_state["analysis_result"] = {
        "dataframe": frame,
        "pre_analysis": pre_analysis,
        "prompt_input": prompt_input,
        "diagnosis": diagnosis,
    }

def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def first_available_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def available_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def build_python_findings(pre_analysis: dict) -> list[str]:
    metrics = pre_analysis["performance_metrics"]
    quality = pre_analysis["quality_checks"]
    checks = pre_analysis["diagnostic_checks"]
    findings: list[str] = []

    findings.append(
        f"Records: {pre_analysis['file_info']['row_count']:,}; columns: {len(pre_analysis['file_info']['column_names'])}; grain: {pre_analysis['file_understanding']['data_grain']}."
    )

    if metrics.get("otp_percent") is not None:
        findings.append(
            f"OTP is {metrics['otp_percent']:.1f}% with average delay {metrics.get('avg_delay', 0):.2f} minutes and max delay {metrics.get('max_delay', 0):.1f} minutes."
        )

    if metrics.get("trip_count") or metrics.get("station_count"):
        findings.append(
            f"Unique trips: {metrics.get('trip_count') or 'n/a'}; unique stations: {metrics.get('station_count') or 'n/a'}."
        )

    if quality["duplicate_rows"] > 0:
        findings.append(f"Duplicate rows detected: {quality['duplicate_rows']}.")
    if quality["duplicate_key_rows"] > 0:
        findings.append(
            f"Duplicate key combinations detected: {quality['duplicate_key_rows']} on {', '.join(quality['duplicate_key_columns'])}."
        )
    if quality["missing_value_rate"] > 0:
        findings.append(f"Missing-value rate: {quality['missing_value_rate']:.1%}.")
    if quality["bad_datetime_rate"] > 0:
        findings.append(f"Bad datetime parsing rate: {quality['bad_datetime_rate']:.1%}.")

    if checks["join_mapping_quality"]["score"] >= 0.5:
        findings.append(checks["join_mapping_quality"]["evidence"][0] + ".")
    if checks["missing_records"]["score"] >= 0.4:
        findings.append(
            f"{checks['missing_records']['evidence'][1]}; {checks['missing_records']['evidence'][0].lower()}."
        )
    if checks["sensor_anomaly"]["score"] >= 0.4:
        findings.append(
            f"{checks['sensor_anomaly']['evidence'][0]}; largest delays appear isolated rather than broad-based."
        )
    if checks["operational_disruption"]["score"] >= 0.5:
        findings.append(
            f"{checks['operational_disruption']['evidence'][0]}; {checks['operational_disruption']['evidence'][1]}; {checks['operational_disruption']['evidence'][2]}."
        )

    return findings


def format_number(value: object, digits: int = 1, suffix: str = "") -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def format_whole_number(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def build_segment_highlights(pre_analysis: dict, segment: str) -> list[str]:
    segmentation = pre_analysis.get("segmentation_analysis", {})
    records = segmentation.get(segment, [])
    if not records:
        return []

    frame = dataframe_from_records(records)
    if frame.empty:
        return []

    if segment == "by_trip":
        label_col = first_available_column(frame, ["group_value", "trip_id"])
        if not label_col or "avg_delay" not in frame.columns:
            return []
        ranked = frame.sort_values(
            ["avg_delay"] + (["record_count"] if "record_count" in frame.columns else []),
            ascending=[False] + ([False] if "record_count" in frame.columns else []),
        ).head(3)
        highlights = []
        for _, row in ranked.iterrows():
            text = (
                f"{row[label_col]} averages {format_number(row.get('avg_delay'), 2, ' min')} delay"
                f" across {format_whole_number(row.get('record_count'))} records"
            )
            if "otp_percent" in ranked.columns:
                text += f" with OTP at {format_number(row.get('otp_percent'), 1, '%')}"
            if "missing_stops" in ranked.columns and row.get("missing_stops", 0):
                text += f" and {format_whole_number(row.get('missing_stops'))} missing stops"
            highlights.append(text + ".")
        return highlights

    if segment == "by_station":
        label_col = first_available_column(frame, ["group_value", "stop_id"])
        if not label_col or "avg_delay" not in frame.columns:
            return []
        ranked = frame.sort_values(
            ["avg_delay"] + (["record_count"] if "record_count" in frame.columns else []),
            ascending=[False] + ([False] if "record_count" in frame.columns else []),
        ).head(3)
        return [
            (
                f"{row[label_col]} shows {format_number(row.get('avg_delay'), 2, ' min')} average delay"
                f" over {format_whole_number(row.get('record_count'))} rows"
                + (
                    f" with OTP at {format_number(row.get('otp_percent'), 1, '%')}"
                    if "otp_percent" in ranked.columns
                    else ""
                )
                + "."
            )
            for _, row in ranked.iterrows()
        ]

    if segment == "by_hour":
        if "hour" not in frame.columns or "avg_delay" not in frame.columns:
            return []
        ranked = frame.sort_values(
            ["avg_delay"] + (["record_count"] if "record_count" in frame.columns else []),
            ascending=[False] + ([False] if "record_count" in frame.columns else []),
        ).head(3)
        return [
            (
                f"{int(row['hour'])}:00 has {format_number(row.get('avg_delay'), 2, ' min')} average delay"
                f" across {format_whole_number(row.get('record_count'))} records"
                + (
                    f" with OTP at {format_number(row.get('otp_percent'), 1, '%')}"
                    if "otp_percent" in ranked.columns
                    else ""
                )
                + "."
            )
            for _, row in ranked.iterrows()
        ]

    if segment == "by_direction":
        label_col = first_available_column(frame, ["direction"])
        if not label_col or "avg_delay" not in frame.columns:
            return []
        ranked = frame.sort_values(
            ["avg_delay"] + (["record_count"] if "record_count" in frame.columns else []),
            ascending=[False] + ([False] if "record_count" in frame.columns else []),
        ).head(3)
        return [
            (
                f"{row[label_col]} direction averages {format_number(row.get('avg_delay'), 2, ' min')} delay"
                + (
                    f" with OTP at {format_number(row.get('otp_percent'), 1, '%')}"
                    if "otp_percent" in ranked.columns
                    else ""
                )
                + "."
            )
            for _, row in ranked.iterrows()
        ]

    return []


def render_segment_card(title: str, items: list[str]) -> None:
    if not items:
        return
    st.markdown(
        """
        <div class="list-card">
            <div class="section-title">"""
        + title
        + """</div>
            <ul>
        """
        + "".join(f"<li>{item}</li>" for item in items)
        + """
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_to_excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="raw_data")
    buffer.seek(0)
    return buffer.getvalue()


def styled_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.68)",
        margin=dict(l=20, r=20, t=68, b=24),
        font=dict(color="#e2e8f0", size=13),
        title=dict(x=0.02, xanchor="left", font=dict(size=18, color="#f8fafc")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#cbd5e1"),
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.96)",
            bordercolor="rgba(96,165,250,0.35)",
            font=dict(color="#f8fafc", size=12),
        ),
        bargap=0.24,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#94a3b8"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.18)",
        tickfont=dict(color="#cbd5e1"),
        title_font=dict(color="#94a3b8"),
    )
    return fig

def render_station_hour_heatmap(chart_data: dict) -> bool:
    station_hour = dataframe_from_records(chart_data.get("station_hour_heatmap", []))
    if station_hour.empty:
        return False
    required = {"stop_id", "hour", "avg_delay"}
    if not required.issubset(station_hour.columns):
        return False

    station_order = (
        station_hour.groupby("stop_id")["avg_delay"].mean().sort_values(ascending=True).index.tolist()
    )
    hour_order = sorted(station_hour["hour"].dropna().astype(int).unique().tolist())

    fig = px.density_heatmap(
        station_hour,
        x="hour",
        y="stop_id",
        z="avg_delay",
        histfunc="avg",
        category_orders={"stop_id": station_order, "hour": hour_order},
        color_continuous_scale=CHART_PALETTE["heatmap_scale_stops"],
        title="Where Delays Are Concentrated by Station and Time",
    )
    fig.update_traces(
        hovertemplate=(
            "Hour: %{x}:00<br>"
            "Station: %{y}<br>"
            "Average delay: %{z:.2f} min<br>"
            "<extra></extra>"
        )
    )
    fig.update_xaxes(title="Hour of day", dtick=1)
    fig.update_yaxes(title="")
    fig.update_coloraxes(colorbar_title="Avg delay")
    st.plotly_chart(styled_figure(fig, height=430), use_container_width=True)
    return True


def render_focus_trip_chart(chart_data: dict) -> bool:
    focus_profiles = dataframe_from_records(chart_data.get("focus_trip_profiles", []))
    if focus_profiles.empty:
        return False

    fig = px.line(
        focus_profiles,
        x="stop_sequence",
        y="delay_min",
        color="trip_id",
        markers=True,
        title="Trips With the Biggest Delay Changes",
        color_discrete_sequence=CHART_PALETTE["comparison"],
        hover_data=["stop_id", "stop_label"],
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "Trip: %{fullData.name}<br>"
            "Stop sequence: %{x}<br>"
            "Delay: %{y:.1f} min<br>"
            "Stop: %{customdata[0]}<br>"
            "<extra></extra>"
        ),
    )
    fig.update_xaxes(dtick=1, title="Stop sequence")
    fig.update_yaxes(title="Delay (min)")
    fig.add_hline(
        y=5,
        line_dash="dash",
        line_color=CHART_PALETTE["threshold"],
        annotation_text="5 min OTP threshold",
        annotation_position="top left",
    )
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_delay_jump_chart(chart_data: dict) -> bool:
    delay_jumps = dataframe_from_records(chart_data.get("delay_jump_events", []))
    if delay_jumps.empty:
        return False

    delay_jumps = delay_jumps.head(8).copy()
    delay_jumps = delay_jumps.sort_values("delay_jump_min", ascending=True)

    fig = px.bar(
        delay_jumps,
        x="delay_jump_min",
        y="segment_label",
        orientation="h",
        title="Biggest Delay Jumps Between Stops",
        color="severity",
        color_discrete_map=SEVERITY_COLOR_MAP,
        hover_data=["prev_stop_id", "current_stop_id", "prev_delay", "current_delay"],
    )
    fig.update_traces(
        texttemplate="%{x:.1f} min",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Jump: %{x:.1f} min<br>"
            "Previous delay: %{customdata[2]:.1f} min<br>"
            "Current delay: %{customdata[3]:.1f} min<br>"
            "<extra></extra>"
        ),
    )
    fig.update_xaxes(title="Increase in delay (min)")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_match_status_chart(chart_data: dict) -> bool:
    match_status = dataframe_from_records(chart_data.get("match_status_breakdown", []))
    if match_status.empty:
        return False

    status_col = first_available_column(match_status, ["match_status"])
    if not status_col or "count" not in match_status.columns:
        return False

    problematic = match_status[
        ~match_status[status_col].astype(str).str.strip().str.lower().eq("matched")
    ].copy()
    problematic = problematic[problematic["count"] > 0]
    if problematic.empty:
        return False

    problematic = problematic.sort_values("count", ascending=True).copy()

    fig = px.bar(
        problematic,
        x="count",
        y=status_col,
        orientation="h",
        title="Records With Match Problems",
        color="share_percent",
        color_continuous_scale=CHART_PALETTE["alert_scale"],
        text="share_percent",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "Status: %{y}<br>"
            "Rows: %{x}<br>"
            "Share: %{marker.color:.1f}%<br>"
            "<extra></extra>"
        ),
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Rows")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_avg_delay_trip_chart(chart_data: dict) -> bool:
    avg_trip = dataframe_from_records(chart_data.get("avg_delay_by_trip", []))
    if avg_trip.empty:
        return False

    label_col = first_available_column(avg_trip, ["trip_id", "group_value"])
    if not label_col or "avg_delay" not in avg_trip.columns:
        return False

    trip_sort_cols = ["avg_delay"] + (["record_count"] if "record_count" in avg_trip.columns else [])
    trip_sort_order = [False] * len(trip_sort_cols)
    avg_trip = avg_trip.sort_values(trip_sort_cols, ascending=trip_sort_order).head(12)
    avg_trip = avg_trip.sort_values("avg_delay", ascending=True)
    hover_cols = available_columns(avg_trip, ["record_count", "otp_percent", "outlier_count"])
    fig = px.bar(
        avg_trip,
        x="avg_delay",
        y=label_col,
        orientation="h",
        title="Trips With the Highest Average Delay",
        color="avg_delay",
        color_continuous_scale=CHART_PALETTE["core_scale"],
        hover_data=hover_cols,
    )
    hover_lines = [
        "Trip: %{y}",
        "Average delay: %{x:.2f} min",
    ]
    if "record_count" in hover_cols:
        hover_lines.append(f"Rows: %{{customdata[{hover_cols.index('record_count')}]}}")
    if "otp_percent" in hover_cols:
        otp_index = hover_cols.index("otp_percent")
        hover_lines.append(f"OTP: %{{customdata[{otp_index}]:.1f}}%")
    if "outlier_count" in hover_cols:
        outlier_index = hover_cols.index("outlier_count")
        hover_lines.append(f"Outliers: %{{customdata[{outlier_index}]}}")
    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    fig.update_traces(hovertemplate="<br>".join(hover_lines) + "<br><extra></extra>")
    fig.add_vline(x=5, line_dash="dash", line_color=CHART_PALETTE["threshold"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Average delay (min)")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_avg_delay_station_chart(chart_data: dict) -> bool:
    avg_station = dataframe_from_records(chart_data.get("avg_delay_by_station", []))
    if avg_station.empty:
        return False

    label_col = first_available_column(avg_station, ["stop_id", "group_value"])
    if not label_col or "avg_delay" not in avg_station.columns:
        return False

    station_sort_cols = ["avg_delay"] + (["record_count"] if "record_count" in avg_station.columns else [])
    station_sort_order = [False] * len(station_sort_cols)
    avg_station = avg_station.sort_values(station_sort_cols, ascending=station_sort_order).head(12)
    avg_station = avg_station.sort_values("avg_delay", ascending=True)
    fig = px.bar(
        avg_station,
        x="avg_delay",
        y=label_col,
        orientation="h",
        title="Stations With the Highest Average Delay",
        color="avg_delay",
        color_continuous_scale=CHART_PALETTE["core_scale"],
        hover_data=available_columns(avg_station, ["record_count", "otp_percent", "outlier_count"]),
    )
    hover_cols = available_columns(avg_station, ["record_count", "otp_percent", "outlier_count"])
    hover_lines = [
        "Station: %{y}",
        "Average delay: %{x:.2f} min",
    ]
    if "record_count" in hover_cols:
        hover_lines.append(f"Rows: %{{customdata[{hover_cols.index('record_count')}]}}")
    if "otp_percent" in hover_cols:
        hover_lines.append(f"OTP: %{{customdata[{hover_cols.index('otp_percent')}]:.1f}}%")
    if "outlier_count" in hover_cols:
        hover_lines.append(f"Outliers: %{{customdata[{hover_cols.index('outlier_count')}]}}")
    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    fig.update_traces(hovertemplate="<br>".join(hover_lines) + "<br><extra></extra>")
    fig.add_vline(x=5, line_dash="dash", line_color=CHART_PALETTE["threshold"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Average delay (min)")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_avg_delay_hour_chart(chart_data: dict) -> bool:
    avg_hour = dataframe_from_records(chart_data.get("avg_delay_by_hour", []))
    if avg_hour.empty:
        return False

    x_col = "time_label" if "time_label" in avg_hour.columns else "hour"
    title = "Average Delay Over Time (15-Minute Windows)" if x_col == "time_label" else "Average Delay Over Time"
    fig = px.line(
        avg_hour,
        x=x_col,
        y="avg_delay",
        markers=True,
        title=title,
    )
    fig.update_traces(
        line=dict(color=CHART_PALETTE["core_scale"][2], width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.10)",
        hovertemplate=(
            ("Window: %{x}<br>" if x_col == "time_label" else "Hour: %{x}:00<br>")
            + "Average delay: %{y:.2f} min<br>"
            + "<extra></extra>"
        ),
    )
    if "is_spike" in avg_hour.columns:
        spike_hours = avg_hour[avg_hour["is_spike"]]
        if not spike_hours.empty:
            fig.add_trace(
                go.Scatter(
                    x=spike_hours[x_col],
                    y=spike_hours["avg_delay"],
                    mode="markers",
                    marker=dict(size=12, color=CHART_PALETTE["spike"], symbol="diamond"),
                    name="Spike window" if x_col == "time_label" else "Spike hour",
                )
            )
    fig.add_hline(x0=0, x1=1, y=5, line_dash="dash", line_color=CHART_PALETTE["threshold"])
    if x_col == "time_label":
        fig.update_xaxes(title="15-minute interval", type="category")
    else:
        fig.update_xaxes(title="Hour of day", dtick=1)
    fig.update_yaxes(title="Average delay (min)")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_otp_trip_chart(chart_data: dict) -> bool:
    otp_trip = dataframe_from_records(chart_data.get("otp_by_trip", []))
    if otp_trip.empty:
        return False

    label_col = first_available_column(otp_trip, ["trip_id", "group_value"])
    if not label_col or "otp_percent" not in otp_trip.columns:
        return False

    otp_sort_cols = ["otp_percent"] + (["record_count"] if "record_count" in otp_trip.columns else [])
    otp_sort_order = [True] + ([False] if "record_count" in otp_trip.columns else [])
    otp_trip = otp_trip.sort_values(otp_sort_cols, ascending=otp_sort_order).head(12)
    otp_trip = otp_trip.sort_values("otp_percent", ascending=True)
    hover_cols = available_columns(otp_trip, ["record_count", "avg_delay", "outlier_count"])

    fig = px.bar(
        otp_trip,
        x="otp_percent",
        y=label_col,
        orientation="h",
        title="Trips With the Lowest On-Time Performance",
        color="otp_percent",
        color_continuous_scale=CHART_PALETTE["otp_scale"],
        hover_data=hover_cols,
    )
    hover_lines = [
        "Trip: %{y}",
        "OTP: %{x:.1f}%",
    ]
    if "record_count" in hover_cols:
        hover_lines.append(f"Rows: %{{customdata[{hover_cols.index('record_count')}]}}")
    if "avg_delay" in hover_cols:
        hover_lines.append(f"Average delay: %{{customdata[{hover_cols.index('avg_delay')}]:.2f}} min")
    if "outlier_count" in hover_cols:
        hover_lines.append(f"Outliers: %{{customdata[{hover_cols.index('outlier_count')}]}}")

    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_traces(hovertemplate="<br>".join(hover_lines) + "<br><extra></extra>")
    fig.add_vline(x=90, line_dash="dash", line_color=CHART_PALETTE["threshold"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="OTP (%)", range=[0, 100])
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_otp_station_chart(chart_data: dict) -> bool:
    otp_station = dataframe_from_records(chart_data.get("otp_by_station", []))
    if otp_station.empty:
        return False

    label_col = first_available_column(otp_station, ["stop_id", "group_value"])
    if not label_col or "otp_percent" not in otp_station.columns:
        return False

    otp_sort_cols = ["otp_percent"] + (["record_count"] if "record_count" in otp_station.columns else [])
    otp_sort_order = [True] + ( [False] if "record_count" in otp_station.columns else [] )
    otp_station = otp_station.sort_values(otp_sort_cols, ascending=otp_sort_order).head(12)
    otp_station = otp_station.sort_values("otp_percent", ascending=True)
    fig = px.bar(
        otp_station,
        x="otp_percent",
        y=label_col,
        orientation="h",
        title="Stations With the Lowest On-Time Performance",
        color="otp_percent",
        color_continuous_scale=CHART_PALETTE["otp_scale"],
        hover_data=available_columns(otp_station, ["record_count", "avg_delay", "outlier_count"]),
    )
    hover_cols = available_columns(otp_station, ["record_count", "avg_delay", "outlier_count"])
    hover_lines = [
        "Station: %{y}",
        "OTP: %{x:.1f}%",
    ]
    if "record_count" in hover_cols:
        hover_lines.append(f"Rows: %{{customdata[{hover_cols.index('record_count')}]}}")
    if "avg_delay" in hover_cols:
        hover_lines.append(f"Average delay: %{{customdata[{hover_cols.index('avg_delay')}]:.2f}} min")
    if "outlier_count" in hover_cols:
        hover_lines.append(f"Outliers: %{{customdata[{hover_cols.index('outlier_count')}]}}")
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_traces(hovertemplate="<br>".join(hover_lines) + "<br><extra></extra>")
    fig.add_vline(x=90, line_dash="dash", line_color=CHART_PALETTE["threshold"])
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="OTP (%)", range=[0, 100])
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_missing_stop_chart(chart_data: dict) -> bool:
    missing_stops = dataframe_from_records(chart_data.get("missing_stop_sequence_summary", []))
    if missing_stops.empty:
        return False
    missing_stops = missing_stops[missing_stops["missing_stops"] > 0]
    if missing_stops.empty:
        return False
    missing_stops = missing_stops.head(12).sort_values("missing_stops", ascending=True)
    fig = px.bar(
        missing_stops,
        x="missing_stops",
        y="trip_id",
        orientation="h",
        title="Trips With Missing Stops",
        color="missing_stops",
        color_continuous_scale=CHART_PALETTE["alert_scale"],
        hover_data=["observed_stops", "expected_stops"],
    )
    fig.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        hovertemplate=(
            "Trip: %{y}<br>"
            "Missing stops: %{x}<br>"
            "Observed: %{customdata[0]}<br>"
            "Expected: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Missing stop count")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_missing_values_chart(pre_analysis: dict) -> bool:
    missing_values = dataframe_from_records(pre_analysis["quality_checks"].get("missing_values", []))
    if missing_values.empty:
        return False
    missing_values = missing_values[missing_values["missing_count"] > 0]
    if missing_values.empty:
        return False
    missing_values = missing_values.sort_values("missing_count", ascending=True).tail(10)
    fig = px.bar(
        missing_values,
        x="missing_count",
        y="column",
        orientation="h",
        title="Columns With Missing Records",
        color="missing_percent",
        color_continuous_scale=CHART_PALETTE["alert_scale"],
    )
    fig.update_traces(
        texttemplate="%{x}",
        textposition="outside",
        hovertemplate=(
            "Column: %{y}<br>"
            "Missing rows: %{x}<br>"
            "Missing rate: %{marker.color:.1f}%<br>"
            "<extra></extra>"
        ),
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="Missing rows")
    fig.update_yaxes(title="")
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_delay_distribution_chart(chart_data: dict) -> bool:
    delay_distribution = dataframe_from_records(chart_data.get("delay_distribution", []))
    if delay_distribution.empty:
        return False
    if {"bin_start", "bin_end"}.issubset(delay_distribution.columns):
        delay_distribution["severity_midpoint"] = (
            delay_distribution["bin_start"] + delay_distribution["bin_end"]
        ) / 2
    else:
        delay_distribution["severity_midpoint"] = delay_distribution["count"]
    fig = px.bar(
        delay_distribution,
        x="label",
        y="count",
        title="How Delay Is Distributed Across Records",
        color="severity_midpoint",
        color_continuous_scale=CHART_PALETTE["distribution_scale_stops"],
    )
    fig.update_traces(
        hovertemplate=(
            "Delay band: %{x}<br>"
            "Rows: %{y}<br>"
            "<extra></extra>"
        )
    )
    fig.update_xaxes(title="Delay band (min)")
    fig.update_yaxes(title="Rows")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(styled_figure(fig), use_container_width=True)
    return True


def render_visual_section(title: str, description: str, panels: list[callable]) -> int:
    section_placeholder = st.empty()
    section_placeholder.markdown(
        f"""
        <div class="section-card" style="padding-bottom:0.6rem;">
            <div class="section-title">{title}</div>
            <div class="insight-copy" style="font-size:0.96rem;color:#cbd5e1;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rendered = 0
    row_columns = st.columns(2)
    column_index = 0
    for panel in panels:
        with row_columns[column_index]:
            success = panel()
        if success:
            rendered += 1
            column_index = 1 - column_index
            if column_index == 0:
                row_columns = st.columns(2)
    if rendered == 0:
        section_placeholder.empty()
    return rendered


def render_visuals(pre_analysis: dict, diagnosis: dict) -> None:
    chart_data = pre_analysis["chart_ready_outputs"]

    st.markdown(
        f"""
        <div class="pill-note" style="margin-bottom:1rem;">
            Visual diagnostics are organized as an analyst workflow: first locate the network pattern, then inspect station impact, then inspect train-level behavior, then check for feed or matching artifacts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    rendered_total = 0
    rendered_total += render_visual_section(
        "System Pattern",
        "These views answer whether OTP deterioration is broad or isolated, and whether it appears at a particular time of day or across the whole network.",
        [
            lambda: render_station_hour_heatmap(chart_data),
            lambda: render_avg_delay_hour_chart(chart_data),
            lambda: render_delay_distribution_chart(chart_data),
        ],
    )
    rendered_total += render_visual_section(
        "Station Impact",
        "Use these views to identify where riders experience the problem most clearly. Stations are usually the right first cut for transit performance review.",
        [
            lambda: render_otp_station_chart(chart_data),
            lambda: render_avg_delay_station_chart(chart_data),
        ],
    )
    rendered_total += render_visual_section(
        "Train Impact",
        "Use these views to see whether the issue is concentrated in a few trains, whether it builds gradually, or whether it jumps abruptly within specific runs.",
        [
            lambda: render_avg_delay_trip_chart(chart_data),
            lambda: render_otp_trip_chart(chart_data),
            lambda: render_focus_trip_chart(chart_data),
            lambda: render_delay_jump_chart(chart_data),
        ],
    )
    rendered_total += render_visual_section(
        "Data Integrity Checks",
        "These views test whether the apparent delay pattern could be caused by missing rows, bad schedule matching, or other feed-quality issues rather than operations.",
        [
            lambda: render_missing_stop_chart(chart_data),
            lambda: render_missing_values_chart(pre_analysis),
            lambda: render_match_status_chart(chart_data),
        ],
    )

    if rendered_total == 0:
        st.markdown(
            f"""
            <div class="empty-state">
                No chart-ready summaries were available for this file.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_overview(pre_analysis: dict, diagnosis: dict) -> None:
    if diagnosis.get("source") == "vertex_unavailable":
        st.error(diagnosis.get("warning", "Gemini analysis failed."))
        render_metric_grid(pre_analysis, diagnosis)
        return

    analysis = str(diagnosis.get("analysis", "")).strip()
    file_name = pre_analysis["file_info"]["file_name"]
    scenario_title = SAMPLE_DESCRIPTIONS.get(
        file_name,
        {"title": file_name.replace(".csv", "").replace(".xlsx", "").replace("_", " ").title()},
    )["title"]
    st.markdown(
        f"""
        <div class="overview-panel">
            <div class="overview-kicker">What {scenario_title} Tells You</div>
            <div class="overview-summary">{analysis}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_grid(pre_analysis, diagnosis)
    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95])
    with left:
        what_happened = diagnosis.get("what_happened", [])[:3]
        st.markdown(
            """
            <div class="list-card">
                <div class="section-title">What Happened</div>
                <ul>
            """
            + "".join(f"<li>{item}</li>" for item in what_happened)
            + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        next_steps = diagnosis["recommended_next_steps"][:3]
        st.markdown(
            """
            <div class="list-card">
                <div class="section-title">Recommended Next Steps</div>
                <ul>
            """
            + "".join(f"<li>{item}</li>" for item in next_steps)
            + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_raw_data(frame: pd.DataFrame, file_name: str, prompt_input: str | None = None) -> None:
    export_name = Path(file_name).stem + "_raw_data.xlsx"
    st.download_button(
        "Download As Excel",
        data=dataframe_to_excel_bytes(frame),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )
    if prompt_input:
        with st.expander("Prompt Input Sent To Gemini"):
            escaped_prompt = (
                str(prompt_input)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            st.markdown(
                f"""
                <div class="prompt-preview">
                    <pre>{escaped_prompt}</pre>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.dataframe(frame.head(100), use_container_width=True, hide_index=True)


def main() -> None:
    inject_css()
    render_hero()

    selected_sample_path, analyze_requested = render_sample_selector()

    active_file, active_name = get_active_file(selected_sample_path)
    if active_name:
        st.markdown(
            f"""
            <div class="active-file-card">
                <div class="active-file-label">Current Input</div>
                <div class="active-file-name">{active_name}</div>
                <div class="active-file-meta">Sample dataset selected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_requested:
        if active_file is None or active_name is None:
            st.error("Choose a sample dataset first.")
        else:
            run_analysis(active_file, active_name)

    result = st.session_state.get("analysis_result")
    if not result:
        st.markdown(
            """
            <div class="empty-state">
                Select a scenario card above to run the analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    tabs = st.tabs(["Overview", "Visual Diagnostics", "Raw Data"])
    with tabs[0]:
        render_overview(result["pre_analysis"], result["diagnosis"])
    with tabs[1]:
        render_visuals(result["pre_analysis"], result["diagnosis"])
    with tabs[2]:
        render_raw_data(
            result["dataframe"],
            result["pre_analysis"]["file_info"]["file_name"],
            result.get("prompt_input"),
        )


if __name__ == "__main__":
    main()
