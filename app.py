from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gemini_helper import run_vertex_diagnosis
from pre_analysis import load_dataframe, run_pre_analysis
from prompts import make_json_safe


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
ISSUE_META = {
    "data_quality": {"label": "Data Quality", "class": "badge-quality"},
    "measurement_logic": {"label": "Measurement Logic", "class": "badge-logic"},
    "real_operations": {"label": "Real Operations", "class": "badge-ops"},
    "insufficient_evidence": {"label": "Insufficient Evidence", "class": "badge-evidence"},
}
SAMPLE_DESCRIPTIONS = {
    "01_sensor_anomaly.csv": "Isolated extreme delays with otherwise normal service.",
    "02_service_pattern_mismatch.csv": "Repeated mid-line spikes that resemble route or stop mismatch behavior.",
    "03_join_mapping_issue.csv": "Schedule matching quality degrades and delays become broadly distorted.",
    "04_missing_records.csv": "A partial feed where rows disappear for some trip windows.",
    "05_true_operational_disruption.csv": "Sustained downstream delay across many trips and stops.",
}


st.set_page_config(
    page_title="Subway OTP Diagnostics",
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
                radial-gradient(circle at top right, rgba(22, 163, 74, 0.14), transparent 26%),
                linear-gradient(180deg, #0a0f1c 0%, #090d18 100%);
            color: #edf2ff;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
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
            color: #93c5fd;
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
            max-width: 820px;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .hero-chip {
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            color: #dbeafe;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.9rem;
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
            color: #93c5fd;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 0.2rem 0;
        }
        .metric-card {
            border-radius: 22px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.82));
            border: 1px solid rgba(148, 163, 184, 0.12);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .metric-label {
            color: #94a3b8;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 0.3rem;
            color: #f8fafc;
        }
        .metric-caption {
            color: #cbd5e1;
            font-size: 0.82rem;
            margin-top: 0.3rem;
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
        .sample-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .sample-copy {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.8rem;
            margin: 0.8rem 0 0.4rem 0;
        }
        .issue-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .badge-quality {
            background: rgba(244, 114, 182, 0.14);
            color: #f9a8d4;
            border: 1px solid rgba(244, 114, 182, 0.26);
        }
        .badge-logic {
            background: rgba(59, 130, 246, 0.14);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.26);
        }
        .badge-ops {
            background: rgba(34, 197, 94, 0.14);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.26);
        }
        .badge-evidence {
            background: rgba(250, 204, 21, 0.14);
            color: #fde68a;
            border: 1px solid rgba(250, 204, 21, 0.26);
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
            margin: 0.1rem 0 0 1.1rem;
        }
        .list-card li {
            color: #dbe4f0;
            margin-bottom: 0.55rem;
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
        .stFileUploader > div {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 18px;
            border: 1px dashed rgba(148, 163, 184, 0.25);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.72);
            border-radius: 16px;
            padding: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            color: #cbd5e1;
            padding: 0.7rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(59, 130, 246, 0.15);
            color: #eff6ff;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Premium Diagnostic Console</div>
            <h1 class="hero-title">Diagnose subway OTP issues with structured evidence and Vertex AI.</h1>
            <p class="hero-copy">
                Upload a CSV or Excel file, run a clean Python pre-analysis, and turn the signals into a polished
                operations diagnosis with visuals, KPIs, and executive-ready next steps.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">CSV + XLSX input</span>
                <span class="hero-chip">Pandas pre-analysis</span>
                <span class="hero-chip">Vertex AI explanation</span>
                <span class="hero-chip">Presentation-ready charts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


def render_metric_grid(pre_analysis: dict, diagnosis: dict) -> None:
    metrics = pre_analysis["performance_metrics"]
    cards = [
        metric_card("Selected File", pre_analysis["file_info"]["file_name"], "Current dataset in focus"),
        metric_card("Rows", f"{pre_analysis['file_info']['row_count']:,}", "Records analyzed"),
        metric_card("OTP", f"{metrics.get('otp_percent', 0) or 0:.1f}%", "Using delay <= 5 min"),
        metric_card("Avg Delay", f"{metrics.get('avg_delay', 0) or 0:.2f} min", "Across valid delay rows"),
        metric_card("Outliers", f"{metrics.get('outlier_count', 0):,}", "IQR-based delay outliers"),
        metric_card(
            "Likely Issue",
            ISSUE_META.get(diagnosis["likely_issue_type"], ISSUE_META["insufficient_evidence"])["label"],
            "Most supported diagnosis",
        ),
    ]
    st.markdown(f"<div class='metric-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_sample_selector() -> Path | None:
    st.markdown("<div class='section-title'>Choose A Sample Dataset</div>", unsafe_allow_html=True)
    sample_files = sorted(SAMPLE_DIR.glob("*"))
    if not sample_files:
        st.info("No sample files found in sample_data.")
        return None

    selected = st.session_state.get("selected_sample", sample_files[0].name)
    columns = st.columns(3)
    for index, sample_file in enumerate(sample_files):
        with columns[index % 3]:
            is_selected = selected == sample_file.name
            selected_class = "selected" if is_selected else ""
            st.markdown(
                f"""
                <div class="sample-card {selected_class}">
                    <div class="sample-title">{sample_file.stem.replace('_', ' ').title()}</div>
                    <div class="sample-copy">{SAMPLE_DESCRIPTIONS.get(sample_file.name, 'Built-in scenario dataset.')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Use this sample",
                key=f"sample-{sample_file.name}",
                use_container_width=True,
            ):
                st.session_state["selected_sample"] = sample_file.name
                selected = sample_file.name

    return SAMPLE_DIR / selected


def get_active_file(sample_path: Path | None, uploaded_file) -> tuple[object | None, str | None]:
    if uploaded_file is not None:
        return uploaded_file, uploaded_file.name
    if sample_path and sample_path.exists():
        return sample_path, sample_path.name
    return None, None


def run_analysis(file_source: object, file_name: str) -> None:
    with st.spinner("Running pre-analysis and preparing the Vertex AI prompt..."):
        frame, metadata = load_dataframe(file_source, file_name)
        pre_analysis = run_pre_analysis(frame, metadata["file_name"])
        diagnosis = run_vertex_diagnosis(pre_analysis)

    st.session_state["analysis_result"] = {
        "dataframe": frame,
        "pre_analysis": pre_analysis,
        "diagnosis": diagnosis,
    }


def build_issue_badge(issue_type: str) -> str:
    meta = ISSUE_META.get(issue_type, ISSUE_META["insufficient_evidence"])
    return f"<span class='issue-badge {meta['class']}'>{meta['label']}</span>"


def dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()


def styled_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def render_visuals(pre_analysis: dict) -> None:
    chart_data = pre_analysis["chart_ready_outputs"]

    chart_one, chart_two = st.columns(2)
    with chart_one:
        delay_distribution = dataframe_from_records(chart_data["delay_distribution"])
        if not delay_distribution.empty:
            fig = px.bar(
                delay_distribution,
                x="label",
                y="count",
                title="Delay Distribution",
                color="count",
                color_continuous_scale=["#0f172a", "#2563eb", "#60a5fa"],
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    with chart_two:
        avg_trip = dataframe_from_records(chart_data["avg_delay_by_trip"])
        if not avg_trip.empty:
            fig = px.bar(
                avg_trip,
                x="trip_id",
                y="avg_delay",
                title="Average Delay by Trip",
                color="avg_delay",
                color_continuous_scale=["#1d4ed8", "#38bdf8", "#7dd3fc"],
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    chart_three, chart_four = st.columns(2)
    with chart_three:
        avg_station = dataframe_from_records(chart_data["avg_delay_by_station"])
        if not avg_station.empty:
            fig = px.bar(
                avg_station,
                x="avg_delay",
                y="stop_id",
                orientation="h",
                title="Average Delay by Station",
                color="avg_delay",
                color_continuous_scale=["#0f766e", "#14b8a6", "#67e8f9"],
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    with chart_four:
        avg_hour = dataframe_from_records(chart_data["avg_delay_by_hour"])
        if not avg_hour.empty:
            fig = px.line(
                avg_hour,
                x="hour",
                y="avg_delay",
                markers=True,
                title="Average Delay by Hour",
            )
            fig.update_traces(line=dict(color="#93c5fd", width=3), marker=dict(size=8))
            if "is_spike" in avg_hour.columns:
                spike_hours = avg_hour[avg_hour["is_spike"]]
                if not spike_hours.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=spike_hours["hour"],
                            y=spike_hours["avg_delay"],
                            mode="markers",
                            marker=dict(size=12, color="#f472b6", symbol="diamond"),
                            name="Spike hour",
                        )
                    )
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    match_status = dataframe_from_records(chart_data["match_status_breakdown"])
    if not match_status.empty:
        fig = px.pie(
            match_status,
            names="match_status",
            values="count",
            title="Match Status Breakdown",
            hole=0.62,
            color_discrete_sequence=["#60a5fa", "#22c55e", "#f59e0b", "#f472b6", "#c084fc"],
        )
        st.plotly_chart(styled_figure(fig, height=420), use_container_width=True)


def render_overview(pre_analysis: dict, diagnosis: dict) -> None:
    metrics = pre_analysis["performance_metrics"]
    issue_type = diagnosis["likely_issue_type"]
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Executive Readout</div>
            <div class="status-strip">
                {build_issue_badge(issue_type)}
                <span class="pill-note">Source: {diagnosis.get('source', 'rule_based').replace('_', ' ').title()}</span>
            </div>
            <p class="insight-copy">{diagnosis['analysis']}</p>
            <span class="pill-note">
                Grain: {pre_analysis['file_understanding']['data_grain']} |
                Trips: {metrics.get('trip_count') or 'n/a'} |
                Stations: {metrics.get('station_count') or 'n/a'}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_grid(pre_analysis, diagnosis)

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            <div class="list-card">
                <div class="section-title">Top Hypotheses</div>
                <ul>
            """
            + "".join(f"<li>{item}</li>" for item in diagnosis["top_hypotheses"])
            + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="list-card">
                <div class="section-title">Recommended Next Steps</div>
                <ul>
            """
            + "".join(f"<li>{item}</li>" for item in diagnosis["recommended_next_steps"])
            + """
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analysis(pre_analysis: dict, diagnosis: dict) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Analysis</div>
            <div class="status-strip">
                {build_issue_badge(diagnosis['likely_issue_type'])}
                <span class="pill-note">Source: {diagnosis.get('source', 'rule_based').replace('_', ' ').title()}</span>
            </div>
            <p class="insight-copy">{diagnosis['analysis']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    issue_scores = diagnosis.get("issue_scores") or pre_analysis["evidence_summary"].get("issue_scores", {})
    if issue_scores:
        scores = pd.DataFrame(
            [{"issue_type": key.replace("_", " ").title(), "score": value} for key, value in issue_scores.items()]
        ).sort_values("score", ascending=False)
        fig = px.bar(
            scores,
            x="score",
            y="issue_type",
            orientation="h",
            title="Pattern Signal Strength",
            color="score",
            color_continuous_scale=["#1e293b", "#3b82f6", "#93c5fd"],
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    if diagnosis.get("warning"):
        st.warning(diagnosis["warning"])


def render_raw_data(pre_analysis: dict, frame: pd.DataFrame) -> None:
    st.dataframe(frame.head(100), use_container_width=True, hide_index=True)
    with st.expander("View raw pre-analysis JSON"):
        st.code(json.dumps(make_json_safe(pre_analysis), indent=2), language="json")


def main() -> None:
    inject_css()
    render_hero()

    selected_sample_path = render_sample_selector()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Upload Your Own File</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop in a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    active_file, active_name = get_active_file(selected_sample_path, uploaded_file)
    if active_name:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Current Input</div>
                <p class="insight-copy"><strong>{active_name}</strong></p>
                <span class="pill-note">{'Uploaded file' if uploaded_file is not None else 'Sample dataset selected'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Run Analysis", type="primary", use_container_width=True):
        if active_file is None or active_name is None:
            st.error("Choose a sample dataset or upload a file first.")
        else:
            run_analysis(active_file, active_name)

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Pick a sample or upload a file, then click Run Analysis to generate the dashboard.")
        return

    tabs = st.tabs(["Overview", "Visual Diagnostics", "Analysis", "Raw Data"])
    with tabs[0]:
        render_overview(result["pre_analysis"], result["diagnosis"])
    with tabs[1]:
        render_visuals(result["pre_analysis"])
    with tabs[2]:
        render_analysis(result["pre_analysis"], result["diagnosis"])
    with tabs[3]:
        render_raw_data(result["pre_analysis"], result["dataframe"])


if __name__ == "__main__":
    main()
