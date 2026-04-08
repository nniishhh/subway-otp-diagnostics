from __future__ import annotations

import io
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


DIMENSION_CANDIDATES = {
    "trip_id": ["trip_id", "trip", "train_id", "run_id"],
    "stop_id": ["stop_id", "station_id", "station", "stop_code"],
    "stop_sequence": ["stop_sequence", "stop_seq", "sequence", "seq"],
    "delay_min": ["delay_min", "delay", "delay_minutes", "minutes_late"],
    "match_status": ["match_status", "match", "status"],
    "direction": ["direction", "dir", "travel_direction"],
    "date": ["date", "service_date"],
    "time": ["time", "scheduled_time"],
    "scheduled_arrival": ["scheduled_arrival", "scheduled_time_full", "scheduled_datetime"],
    "actual_arrival": ["actual_arrival", "actual_time_full", "actual_datetime"],
}
ISSUE_TYPES = [
    "data_quality",
    "measurement_logic",
    "real_operations",
    "insufficient_evidence",
]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def round_value(value: Any, digits: int = 2) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return round(float(value), digits)
    return value


def clean_score(value: Any, digits: int = 2) -> float:
    return round(float(value), digits)


def to_serializable_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    cleaned = frame.copy()
    if limit is not None:
        cleaned = cleaned.head(limit)

    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)
    records = cleaned.to_dict(orient="records")
    output: list[dict[str, Any]] = []
    for record in records:
        output.append({key: round_value(value) for key, value in record.items()})
    return output


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        match = normalized.get(normalize_name(candidate))
        if match:
            return match
    return None


def detect_dimensions(frame: pd.DataFrame) -> dict[str, str | None]:
    columns = list(frame.columns)
    return {
        key: find_column(columns, candidates)
        for key, candidates in DIMENSION_CANDIDATES.items()
    }


def load_dataframe(file_source: Any, file_name: str | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    detected_name = file_name or getattr(file_source, "name", "uploaded_file")
    suffix = Path(detected_name).suffix.lower()

    if hasattr(file_source, "getvalue"):
        payload = file_source.getvalue()
    elif isinstance(file_source, (str, Path)):
        payload = None
    else:
        payload = file_source.read()

    if suffix == ".csv":
        if payload is None:
            frame = pd.read_csv(file_source)
        else:
            frame = pd.read_csv(io.BytesIO(payload))
    elif suffix in {".xlsx", ".xls"}:
        if payload is None:
            frame = pd.read_excel(file_source)
        else:
            frame = pd.read_excel(io.BytesIO(payload))
    else:
        raise ValueError("Please upload a CSV or Excel file.")

    frame.columns = [str(column).strip() for column in frame.columns]
    metadata = {"file_name": detected_name, "file_type": suffix.lstrip(".") or "unknown"}
    return frame, metadata


def infer_data_grain(dimensions: dict[str, str | None]) -> str:
    if dimensions["trip_id"] and dimensions["stop_id"]:
        return "trip-stop level"
    if dimensions["trip_id"]:
        return "trip level"
    if dimensions["stop_id"]:
        return "station level"
    return "record level"


def parse_datetime_features(
    frame: pd.DataFrame, dimensions: dict[str, str | None]
) -> tuple[pd.Series | None, dict[str, int]]:
    issues: dict[str, int] = {}

    for key in ["actual_arrival", "scheduled_arrival"]:
        column = dimensions.get(key)
        if column:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            issues[key] = int(parsed.isna().sum() - frame[column].isna().sum())
            if parsed.notna().any():
                return parsed, issues

    date_column = dimensions.get("date")
    time_column = dimensions.get("time")
    if date_column and time_column:
        combined = (
            frame[date_column].astype(str).str.strip()
            + " "
            + frame[time_column].astype(str).str.strip()
        )
        parsed = pd.to_datetime(combined, errors="coerce")
        issues["date_time_combined"] = int(parsed.isna().sum() - combined.isna().sum())
        if parsed.notna().any():
            return parsed, issues

    return None, issues


def compute_outliers(delay: pd.Series | None) -> tuple[pd.Series, dict[str, float]]:
    if delay is None or delay.dropna().shape[0] < 4:
        return pd.Series(False, index=delay.index if delay is not None else []), {
            "lower_bound": 0.0,
            "upper_bound": 0.0,
        }

    valid_delay = delay.dropna()
    q1 = valid_delay.quantile(0.25)
    q3 = valid_delay.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = delay.lt(lower) | delay.gt(upper)
    return outlier_mask.fillna(False), {
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
    }


def compute_missing_stop_sequences(
    frame: pd.DataFrame, trip_col: str | None, stop_sequence_col: str | None
) -> pd.DataFrame:
    if not trip_col or not stop_sequence_col:
        return pd.DataFrame()

    working = frame[[trip_col, stop_sequence_col]].copy()
    working[stop_sequence_col] = pd.to_numeric(working[stop_sequence_col], errors="coerce")
    working = working.dropna(subset=[trip_col, stop_sequence_col])

    if working.empty:
        return pd.DataFrame()

    rows = []
    for trip_id, group in working.groupby(trip_col):
        unique_stops = sorted(set(group[stop_sequence_col].astype(int)))
        expected = unique_stops[-1] - unique_stops[0] + 1
        missing = max(expected - len(unique_stops), 0)
        rows.append(
            {
                "trip_id": trip_id,
                "observed_stops": len(unique_stops),
                "expected_stops": expected,
                "missing_stops": missing,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_stops", "trip_id"], ascending=[False, True]
    )


def compute_delay_distribution(delay: pd.Series | None) -> list[dict[str, Any]]:
    if delay is None or delay.dropna().empty:
        return []

    valid_delay = delay.dropna()
    bin_count = min(12, max(6, int(len(valid_delay) ** 0.5)))
    bins = pd.cut(valid_delay, bins=bin_count)
    counts = bins.value_counts(sort=False)
    rows = []
    for interval, count in counts.items():
        start = float(interval.left)
        end = float(interval.right)
        rows.append(
            {
                "bin_start": round(float(start), 2),
                "bin_end": round(float(end), 2),
                "count": int(count),
                "label": f"{start:.1f} to {end:.1f}",
            }
        )
    return rows


def compute_delay_jump_events(
    frame: pd.DataFrame,
    trip_col: str | None,
    stop_sequence_col: str | None,
    delay_col: str | None,
    stop_col: str | None,
) -> pd.DataFrame:
    if not trip_col or not stop_sequence_col or not delay_col:
        return pd.DataFrame()

    working_columns = [trip_col, stop_sequence_col, delay_col]
    if stop_col:
        working_columns.append(stop_col)

    working = frame[working_columns].copy()
    working[stop_sequence_col] = pd.to_numeric(working[stop_sequence_col], errors="coerce")
    working[delay_col] = pd.to_numeric(working[delay_col], errors="coerce")
    working = working.dropna(subset=[trip_col, stop_sequence_col, delay_col])
    if working.empty:
        return pd.DataFrame()

    working = working.sort_values([trip_col, stop_sequence_col]).copy()
    working["prev_delay"] = working.groupby(trip_col)[delay_col].shift(1)
    working["prev_sequence"] = working.groupby(trip_col)[stop_sequence_col].shift(1)
    if stop_col:
        working["prev_stop_id"] = working.groupby(trip_col)[stop_col].shift(1)
        working["current_stop_id"] = working[stop_col]
    else:
        working["prev_stop_id"] = ""
        working["current_stop_id"] = ""

    working["delay_jump_min"] = working[delay_col] - working["prev_delay"]
    working["sequence_gap"] = working[stop_sequence_col] - working["prev_sequence"]

    jump_events = working[
        working["prev_delay"].notna()
        & working["delay_jump_min"].notna()
        & working["sequence_gap"].eq(1)
        & working["delay_jump_min"].gt(0)
    ].copy()

    if jump_events.empty:
        return pd.DataFrame()

    jump_events["severity"] = pd.cut(
        jump_events["delay_jump_min"],
        bins=[0, 3, 6, 10, float("inf")],
        labels=["minor", "moderate", "major", "critical"],
        include_lowest=False,
    ).astype(str)
    jump_events["segment_label"] = jump_events.apply(
        lambda row: (
            f"{row[trip_col]}: {row['prev_stop_id']} -> {row['current_stop_id']}"
            if stop_col
            else f"{row[trip_col]}: seq {int(row['prev_sequence'])} -> {int(row[stop_sequence_col])}"
        ),
        axis=1,
    )

    return jump_events[
        [
            trip_col,
            "prev_stop_id",
            "current_stop_id",
            "prev_delay",
            delay_col,
            "delay_jump_min",
            "severity",
            "segment_label",
        ]
    ].rename(
        columns={
            trip_col: "trip_id",
            delay_col: "current_delay",
        }
    ).sort_values(["delay_jump_min", "trip_id"], ascending=[False, True])


def compute_focus_trip_profiles(
    frame: pd.DataFrame,
    trip_col: str | None,
    stop_sequence_col: str | None,
    delay_col: str | None,
    stop_col: str | None,
    jump_events: pd.DataFrame,
    top_n: int = 4,
) -> pd.DataFrame:
    if not trip_col or not stop_sequence_col or not delay_col:
        return pd.DataFrame()

    if jump_events.empty:
        return pd.DataFrame()

    focus_trips = jump_events["trip_id"].dropna().astype(str).drop_duplicates().head(top_n).tolist()
    if not focus_trips:
        return pd.DataFrame()

    working_columns = [trip_col, stop_sequence_col, delay_col]
    if stop_col:
        working_columns.append(stop_col)

    working = frame[working_columns].copy()
    working = working[working[trip_col].astype(str).isin(focus_trips)].copy()
    if working.empty:
        return pd.DataFrame()

    working[stop_sequence_col] = pd.to_numeric(working[stop_sequence_col], errors="coerce")
    working[delay_col] = pd.to_numeric(working[delay_col], errors="coerce")
    working = working.dropna(subset=[trip_col, stop_sequence_col, delay_col])
    if working.empty:
        return pd.DataFrame()

    working = working.sort_values([trip_col, stop_sequence_col]).copy()
    working["trip_id"] = working[trip_col].astype(str)
    working["stop_sequence"] = working[stop_sequence_col].astype(int)
    working["delay_min"] = working[delay_col]
    if stop_col:
        working["stop_id"] = working[stop_col].astype(str)
        working["stop_label"] = working["stop_id"] + " (" + working["stop_sequence"].astype(str) + ")"
    else:
        working["stop_id"] = ""
        working["stop_label"] = "Seq " + working["stop_sequence"].astype(str)

    return working[["trip_id", "stop_sequence", "stop_id", "stop_label", "delay_min"]]


def compute_duplicate_key_info(
    frame: pd.DataFrame, dimensions: dict[str, str | None]
) -> tuple[list[str], int]:
    key_candidates = [
        [dimensions["trip_id"], dimensions["stop_id"], dimensions["scheduled_arrival"]],
        [dimensions["trip_id"], dimensions["stop_sequence"], dimensions["date"]],
        [dimensions["trip_id"], dimensions["stop_sequence"]],
    ]

    for combo in key_candidates:
        valid_combo = [column for column in combo if column]
        if len(valid_combo) == len(combo) and valid_combo:
            duplicate_count = int(frame.duplicated(subset=valid_combo).sum())
            return valid_combo, duplicate_count

    return [], 0


def build_group_summary(
    frame: pd.DataFrame,
    group_column: str | None,
    delay_column: str | None,
    outlier_mask: pd.Series | None = None,
    stop_gap_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not group_column or not delay_column:
        return pd.DataFrame()

    working = frame[[group_column, delay_column]].copy()
    working[delay_column] = pd.to_numeric(working[delay_column], errors="coerce")
    working["is_on_time"] = working[delay_column].le(5)
    working["is_outlier"] = outlier_mask if outlier_mask is not None else False

    grouped = (
        working.groupby(group_column, dropna=False)
        .agg(
            avg_delay=(delay_column, "mean"),
            otp_percent=("is_on_time", lambda values: 100 * values.mean()),
            record_count=(delay_column, "size"),
            outlier_count=("is_outlier", "sum"),
        )
        .reset_index()
        .rename(columns={group_column: "group_value"})
        .sort_values(["avg_delay", "record_count"], ascending=[False, False])
    )

    if stop_gap_frame is not None and not stop_gap_frame.empty:
        grouped = grouped.merge(
            stop_gap_frame[["trip_id", "missing_stops"]].rename(
                columns={"trip_id": "group_value"}
            ),
            on="group_value",
            how="left",
        )
        grouped["missing_stops"] = grouped["missing_stops"].fillna(0).astype(int)

    return grouped


def build_hour_summary(
    frame: pd.DataFrame,
    parsed_time: pd.Series | None,
    delay_column: str | None,
    outlier_mask: pd.Series | None,
) -> pd.DataFrame:
    if parsed_time is None or delay_column is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        {
            "hour": parsed_time.dt.hour,
            "delay_min": pd.to_numeric(frame[delay_column], errors="coerce"),
            "is_outlier": outlier_mask if outlier_mask is not None else False,
        }
    ).dropna(subset=["hour"])
    working["is_on_time"] = working["delay_min"].le(5)

    summary = (
        working.groupby("hour")
        .agg(
            avg_delay=("delay_min", "mean"),
            otp_percent=("is_on_time", lambda values: 100 * values.mean()),
            record_count=("delay_min", "size"),
            outlier_count=("is_outlier", "sum"),
        )
        .reset_index()
        .sort_values("hour")
    )
    return summary


def build_time_window_summary(
    frame: pd.DataFrame,
    parsed_time: pd.Series | None,
    delay_column: str | None,
    outlier_mask: pd.Series | None,
    freq: str = "15min",
) -> pd.DataFrame:
    if parsed_time is None or delay_column is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        {
            "time_window": parsed_time.dt.floor(freq),
            "delay_min": pd.to_numeric(frame[delay_column], errors="coerce"),
            "is_outlier": outlier_mask if outlier_mask is not None else False,
        }
    ).dropna(subset=["time_window"])
    working["is_on_time"] = working["delay_min"].le(5)

    summary = (
        working.groupby("time_window")
        .agg(
            avg_delay=("delay_min", "mean"),
            otp_percent=("is_on_time", lambda values: 100 * values.mean()),
            record_count=("delay_min", "size"),
            outlier_count=("is_outlier", "sum"),
        )
        .reset_index()
        .sort_values("time_window")
    )
    if summary.empty:
        return summary

    summary["time_label"] = summary["time_window"].dt.strftime("%H:%M")
    return summary


def build_station_hour_summary(
    frame: pd.DataFrame,
    stop_col: str | None,
    parsed_time: pd.Series | None,
    delay_col: str | None,
) -> pd.DataFrame:
    if stop_col is None or parsed_time is None or delay_col is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        {
            "stop_id": frame[stop_col].astype(str),
            "hour": parsed_time.dt.hour,
            "delay_min": pd.to_numeric(frame[delay_col], errors="coerce"),
        }
    ).dropna(subset=["stop_id", "hour"])
    if working.empty:
        return pd.DataFrame()

    working["is_on_time"] = working["delay_min"].le(5)
    summary = (
        working.groupby(["stop_id", "hour"])
        .agg(
            avg_delay=("delay_min", "mean"),
            otp_percent=("is_on_time", lambda values: 100 * values.mean()),
            record_count=("delay_min", "size"),
        )
        .reset_index()
    )
    if summary.empty:
        return summary

    station_order = (
        summary.groupby("stop_id")
        .agg(avg_delay=("avg_delay", "mean"), record_count=("record_count", "sum"))
        .sort_values(["avg_delay", "record_count"], ascending=[False, False])
        .head(10)
        .index
    )
    return summary[summary["stop_id"].isin(station_order)].copy()


def build_direction_summary(
    frame: pd.DataFrame, direction_col: str | None, delay_col: str | None
) -> pd.DataFrame:
    if not direction_col or not delay_col:
        return pd.DataFrame()

    working = frame[[direction_col, delay_col]].copy()
    working[delay_col] = pd.to_numeric(working[delay_col], errors="coerce")
    working["is_on_time"] = working[delay_col].le(5)

    return (
        working.groupby(direction_col, dropna=False)
        .agg(
            avg_delay=(delay_col, "mean"),
            otp_percent=("is_on_time", lambda values: 100 * values.mean()),
            record_count=(delay_col, "size"),
        )
        .reset_index()
        .rename(columns={direction_col: "direction"})
        .sort_values("avg_delay", ascending=False)
    )


def compute_match_status_summary(frame: pd.DataFrame, match_status_col: str | None) -> pd.DataFrame:
    if not match_status_col:
        return pd.DataFrame()

    summary = (
        frame[match_status_col]
        .fillna("missing")
        .astype(str)
        .value_counts(normalize=False, dropna=False)
        .rename_axis("match_status")
        .reset_index(name="count")
    )
    summary["share_percent"] = 100 * summary["count"] / max(len(frame), 1)
    return summary


def build_diagnostic_checks(
    frame: pd.DataFrame,
    dimensions: dict[str, str | None],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    trip_summary: pd.DataFrame,
    station_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    match_status_summary: pd.DataFrame,
    missing_stop_summary: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    row_count = metrics["record_count"]
    avg_delay = metrics.get("avg_delay", 0.0) or 0.0
    max_delay = metrics.get("max_delay", 0.0) or 0.0
    otp_percent = metrics.get("otp_percent", 0.0) or 0.0
    outlier_share = metrics.get("outlier_share", 0.0) or 0.0
    missingness_rate = quality["missing_value_rate"]
    duplicate_row_rate = quality["duplicate_row_rate"]
    invalid_value_rate = quality["invalid_value_rate"]
    bad_datetime_rate = quality["bad_datetime_rate"]

    elevated_trip_share = 0.0
    if not trip_summary.empty:
        elevated_trip_share = float((trip_summary["avg_delay"] > 5).mean())

    elevated_station_share = 0.0
    repeated_station_issue_share = 0.0
    concentrated_station_share = 0.0
    if not station_summary.empty:
        elevated_station_share = float((station_summary["avg_delay"] > 5).mean())
        strong_station_pattern = station_summary[
            (station_summary["avg_delay"] > max(avg_delay + 2, 4))
            & (station_summary["record_count"] >= 3)
        ]
        repeated_station_issue_share = len(strong_station_pattern) / max(len(station_summary), 1)
        concentrated_station_share = float(
            (station_summary["avg_delay"] > max(avg_delay + 1.0, 3.5)).mean()
        )

    gapped_trip_share = 0.0
    if not missing_stop_summary.empty:
        gapped_trip_share = float((missing_stop_summary["missing_stops"] > 0).mean())

    non_matched_share = 0.0
    if not match_status_summary.empty:
        matched_rows = match_status_summary[
            match_status_summary["match_status"].astype(str).str.lower().eq("matched")
        ]["count"].sum()
        non_matched_share = 1 - (matched_rows / max(row_count, 1))

    extreme_outlier_bonus = 0.18 if max_delay >= 15 and otp_percent >= 95 else 0.0
    sensor_score = min(
        1.0, max(outlier_share * 10, 0) * max(0.2, 1 - elevated_trip_share) * (1 if otp_percent >= 80 else 0.7)
    )
    sensor_score = min(
        1.0,
        (
            (outlier_share * 14) + extreme_outlier_bonus
        )
        * max(0.25, 1 - elevated_trip_share)
        * max(0.25, 1 - (gapped_trip_share * 2))
        * (1.0 if non_matched_share < 0.2 else 0.7),
    )
    missing_record_score = min(
        1.0,
        (gapped_trip_share * 2.1)
        + (missingness_rate * 2.2)
        + (duplicate_row_rate * 1.2)
        + (bad_datetime_rate * 1.2),
    )
    join_mapping_score = min(1.0, non_matched_share * 1.9)
    service_pattern_score = min(
        1.0,
        (repeated_station_issue_share * 1.2)
        + (
            concentrated_station_share
            * 1.5
            * max(0.3, 1 - elevated_trip_share)
        )
        + (0.15 if 1.5 <= avg_delay <= 5.5 and otp_percent >= 80 else 0)
        + (0.1 if 0 < non_matched_share < 0.35 else 0),
    )
    operational_base = (
        (elevated_trip_share * 0.45)
        + (elevated_station_share * 0.25)
        + ((100 - otp_percent) / 100 * 0.35)
        + min(avg_delay / 14, 0.25)
    )
    operational_penalty = (non_matched_share * 0.7) + (gapped_trip_share * 0.35)
    operational_score = max(
        0.0,
        min(
            1.0,
            operational_base - operational_penalty,
        ),
    )
    insufficient_score = 0.05
    if not dimensions["delay_min"]:
        insufficient_score = 0.95
    elif row_count < 20:
        insufficient_score = 0.85
    elif not dimensions["trip_id"] and not dimensions["stop_id"]:
        insufficient_score = 0.7

    return {
        "sensor_anomaly": {
            "score": clean_score(sensor_score),
            "triggered": bool(sensor_score >= 0.5),
            "evidence": [
                f"Outlier share: {outlier_share:.1%}",
                f"Broadly elevated trips: {elevated_trip_share:.1%}",
            ],
        },
        "missing_records": {
            "score": clean_score(missing_record_score),
            "triggered": bool(missing_record_score >= 0.5),
            "evidence": [
                f"Missing-value rate: {missingness_rate:.1%}",
                f"Trips with stop-sequence gaps: {gapped_trip_share:.1%}",
                f"Bad datetime parsing rate: {bad_datetime_rate:.1%}",
            ],
        },
        "join_mapping_quality": {
            "score": clean_score(join_mapping_score),
            "triggered": bool(join_mapping_score >= 0.5),
            "evidence": [f"Non-matched share: {non_matched_share:.1%}"],
        },
        "service_pattern_mismatch": {
            "score": clean_score(service_pattern_score),
            "triggered": bool(service_pattern_score >= 0.5),
            "evidence": [
                f"Repeated station-level pattern share: {repeated_station_issue_share:.1%}",
                f"Concentrated station share: {concentrated_station_share:.1%}",
                f"Average delay: {avg_delay:.2f} min",
            ],
        },
        "operational_disruption": {
            "score": clean_score(operational_score),
            "triggered": bool(operational_score >= 0.5),
            "evidence": [
                f"Elevated trip share: {elevated_trip_share:.1%}",
                f"Elevated station share: {elevated_station_share:.1%}",
                f"OTP: {otp_percent:.1f}%",
            ],
        },
        "insufficient_evidence": {
            "score": clean_score(insufficient_score),
            "triggered": bool(insufficient_score >= 0.75),
            "evidence": [
                f"Row count: {row_count}",
                f"Delay column available: {bool(dimensions['delay_min'])}",
            ],
        },
        "quality_pressure": {
            "score": clean_score(
                min(
                    1.0,
                    (missingness_rate * 2.5)
                    + (duplicate_row_rate * 1.6)
                    + (invalid_value_rate * 1.6)
                    + (bad_datetime_rate * 1.4),
                )
            ),
            "triggered": bool(
                (missingness_rate + duplicate_row_rate + invalid_value_rate) >= 0.2
            ),
            "evidence": [
                f"Duplicate row rate: {duplicate_row_rate:.1%}",
                f"Invalid-value rate: {invalid_value_rate:.1%}",
            ],
        },
    }


def build_rule_based_diagnosis(
    metrics: dict[str, Any], diagnostic_checks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    insufficient_score = diagnostic_checks["insufficient_evidence"]["score"]
    issue_scores = {
        "data_quality": max(
            diagnostic_checks["missing_records"]["score"],
            min(
                1.0,
                (diagnostic_checks["missing_records"]["score"] * 0.8)
                + (diagnostic_checks["quality_pressure"]["score"] * 0.2),
            ),
        ),
        "measurement_logic": min(
            1.0,
            max(
                diagnostic_checks["join_mapping_quality"]["score"],
                (diagnostic_checks["sensor_anomaly"]["score"] * 1.0)
                + (diagnostic_checks["service_pattern_mismatch"]["score"] * 0.25),
                (diagnostic_checks["service_pattern_mismatch"]["score"] * 0.95)
                + (diagnostic_checks["sensor_anomaly"]["score"] * 0.15),
            ),
        ),
        "real_operations": diagnostic_checks["operational_disruption"]["score"],
        "insufficient_evidence": insufficient_score,
    }

    if insufficient_score >= 0.85:
        likely_issue_type = "insufficient_evidence"
    else:
        likely_issue_type = max(
            [issue for issue in ISSUE_TYPES if issue != "insufficient_evidence"],
            key=lambda issue: issue_scores[issue],
        )
        if issue_scores[likely_issue_type] < 0.35:
            likely_issue_type = "insufficient_evidence"

    sorted_scores = sorted(issue_scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1]
    runner_up = sorted_scores[1][1]
    confidence = max(0.35, min(0.95, top_score * 0.75 + (top_score - runner_up) * 0.5 + 0.25))

    explanations = {
        "data_quality": (
            "Python found evidence of incomplete or unreliable records. "
            "Taken together, this points more to data quality problems than true service performance."
        ),
        "measurement_logic": (
            "Python found patterns that look more like matching, mapping, or measurement artifacts than a broad operating problem. "
            "Taken together, this points to measurement logic."
        ),
        "real_operations": (
            "Python found delay elevated across a broader portion of the network rather than in a few isolated records. "
            "Taken together, this points to a real operational disruption."
        ),
        "insufficient_evidence": (
            "Python found some signals, but the file does not provide enough reliable structure to support a strong root-cause call yet."
        ),
    }

    next_steps = {
        "data_quality": [
            "Inspect missing values, duplicate keys, and stop-sequence gaps before trusting OTP trends.",
            "Validate the feed window and confirm whether rows were dropped during ingestion.",
            "Re-run the analysis after filling or removing clearly broken records.",
        ],
        "measurement_logic": [
            "Review how records are matched to schedules and whether match_status values are trustworthy.",
            "Inspect station-level repeat patterns and extreme outliers for sensor or mapping errors.",
            "Compare against a known-good subset to isolate logic differences from true delays.",
        ],
        "real_operations": [
            "Check the worst trips, stations, and hours to identify where the disruption concentrated.",
            "Compare the disruption window against service advisories, incidents, or dispatch notes.",
            "Use the high-delay segments to estimate operational impact and rider-facing OTP loss.",
        ],
        "insufficient_evidence": [
            "Provide delay_min plus trip or station identifiers if available.",
            "Include a timestamp or date-time field so patterns can be checked over time.",
            "Use a larger or more complete extract before making a final call.",
        ],
    }

    if likely_issue_type == "data_quality":
        top_hypotheses = [
            "Missing values, stop-sequence gaps, or broken datetime parsing suggest the feed may be incomplete.",
            "Data hygiene issues should be resolved before interpreting OTP as a true service outcome.",
            "Observed delay patterns may be partly caused by dropped or malformed records.",
        ]
    elif likely_issue_type == "measurement_logic":
        top_hypotheses = [
            "The pattern looks concentrated or abrupt rather than broad-based, which is consistent with a logic or matching artifact.",
            "Schedule joins or stop-pattern alignment may be distorting the apparent delay story.",
            "A subset of rows may be driving the anomaly instead of a true network slowdown.",
        ]
    elif likely_issue_type == "real_operations":
        top_hypotheses = [
            "Delay appears elevated across multiple parts of the network rather than in a few isolated records.",
            "The OTP drop is more consistent with a real operating slowdown than with a narrow data artifact.",
            "The highest-impact stations, trains, and hours should be reviewed as the operational concentration points.",
        ]
    else:
        top_hypotheses = [
            "The file does not provide enough reliable structure for a confident root-cause explanation.",
            "More complete identifiers or time context would make the diagnosis stronger.",
        ]

    return {
        "likely_issue_type": likely_issue_type,
        "confidence": round(confidence, 2),
        "short_explanation": explanations[likely_issue_type],
        "top_hypotheses": top_hypotheses,
        "recommended_next_steps": next_steps[likely_issue_type],
        "issue_scores": {key: clean_score(value) for key, value in issue_scores.items()},
        "otp_percent": metrics.get("otp_percent"),
    }


def run_pre_analysis(frame: pd.DataFrame, file_name: str) -> dict[str, Any]:
    working = frame.copy()
    dimensions = detect_dimensions(working)
    grain = infer_data_grain(dimensions)
    parsed_time, datetime_issues = parse_datetime_features(working, dimensions)

    delay_col = dimensions["delay_min"]
    delay_series = None
    if delay_col:
        delay_series = pd.to_numeric(working[delay_col], errors="coerce")

    outlier_mask, outlier_bounds = compute_outliers(delay_series)
    stop_gap_summary = compute_missing_stop_sequences(
        working, dimensions["trip_id"], dimensions["stop_sequence"]
    )
    duplicate_key_columns, duplicate_key_rows = compute_duplicate_key_info(working, dimensions)

    missing_values = (
        working.isna().sum().rename_axis("column").reset_index(name="missing_count")
    )
    missing_values["missing_percent"] = 100 * missing_values["missing_count"] / max(len(working), 1)
    missing_values = missing_values.sort_values("missing_count", ascending=False)

    invalid_values = {}
    invalid_value_count = 0
    if delay_col:
        invalid_delay_parse = int(delay_series.isna().sum() - working[delay_col].isna().sum())
        suspicious_delay = int(delay_series.abs().gt(60).fillna(False).sum())
        invalid_values["delay_parse_failures"] = invalid_delay_parse
        invalid_values["suspicious_delay_values"] = suspicious_delay
        invalid_value_count += invalid_delay_parse + suspicious_delay

    stop_sequence_col = dimensions["stop_sequence"]
    if stop_sequence_col:
        stop_sequence = pd.to_numeric(working[stop_sequence_col], errors="coerce")
        bad_stop_sequence = int(stop_sequence.le(0).fillna(False).sum())
        invalid_values["invalid_stop_sequence_values"] = bad_stop_sequence
        invalid_value_count += bad_stop_sequence

    duplicate_rows = int(working.duplicated().sum())
    missing_value_rate = float(working.isna().sum().sum() / max(working.size, 1))
    duplicate_row_rate = float(duplicate_rows / max(len(working), 1))
    invalid_value_rate = float(invalid_value_count / max(len(working), 1))
    bad_datetime_rate = float(sum(datetime_issues.values()) / max(len(working), 1))

    trip_summary = build_group_summary(
        working,
        dimensions["trip_id"],
        dimensions["delay_min"],
        outlier_mask if delay_col else None,
        stop_gap_summary,
    )
    station_summary = build_group_summary(
        working,
        dimensions["stop_id"],
        dimensions["delay_min"],
        outlier_mask if delay_col else None,
    )
    hour_summary = build_hour_summary(
        working, parsed_time, dimensions["delay_min"], outlier_mask if delay_col else None
    )
    time_window_summary = build_time_window_summary(
        working, parsed_time, dimensions["delay_min"], outlier_mask if delay_col else None
    )
    direction_summary = build_direction_summary(
        working, dimensions["direction"], dimensions["delay_min"]
    )
    station_hour_summary = build_station_hour_summary(
        working, dimensions["stop_id"], parsed_time, dimensions["delay_min"]
    )
    match_status_summary = compute_match_status_summary(working, dimensions["match_status"])
    delay_jump_events = compute_delay_jump_events(
        working,
        dimensions["trip_id"],
        dimensions["stop_sequence"],
        dimensions["delay_min"],
        dimensions["stop_id"],
    )
    focus_trip_profiles = compute_focus_trip_profiles(
        working,
        dimensions["trip_id"],
        dimensions["stop_sequence"],
        dimensions["delay_min"],
        dimensions["stop_id"],
        delay_jump_events,
    )

    metrics = {
        "record_count": int(len(working)),
        "column_count": int(len(working.columns)),
        "trip_count": int(working[dimensions["trip_id"]].nunique()) if dimensions["trip_id"] else None,
        "station_count": int(working[dimensions["stop_id"]].nunique()) if dimensions["stop_id"] else None,
        "otp_percent": round(float(delay_series.le(5).mean() * 100), 2)
        if delay_series is not None and delay_series.notna().any()
        else None,
        "avg_delay": round(float(delay_series.mean()), 2)
        if delay_series is not None and delay_series.notna().any()
        else None,
        "median_delay": round(float(delay_series.median()), 2)
        if delay_series is not None and delay_series.notna().any()
        else None,
        "std_delay": round(float(delay_series.std()), 2)
        if delay_series is not None and delay_series.notna().any()
        else None,
        "max_delay": round(float(delay_series.max()), 2)
        if delay_series is not None and delay_series.notna().any()
        else None,
        "outlier_count": int(outlier_mask.sum()) if delay_series is not None else 0,
        "outlier_share": round(float(outlier_mask.mean()), 4) if delay_series is not None else 0.0,
    }

    quality = {
        "missing_values": to_serializable_records(missing_values),
        "missing_value_rate": round(missing_value_rate, 4),
        "duplicate_rows": duplicate_rows,
        "duplicate_row_rate": round(duplicate_row_rate, 4),
        "duplicate_key_columns": duplicate_key_columns,
        "duplicate_key_rows": duplicate_key_rows,
        "invalid_values": invalid_values,
        "invalid_value_rate": round(invalid_value_rate, 4),
        "bad_datetime_parsing": datetime_issues,
        "bad_datetime_rate": round(bad_datetime_rate, 4),
        "outlier_bounds": outlier_bounds,
    }

    diagnostic_checks = build_diagnostic_checks(
        working,
        dimensions,
        metrics,
        quality,
        trip_summary,
        station_summary,
        hour_summary,
        match_status_summary,
        stop_gap_summary,
    )
    evidence_summary = build_rule_based_diagnosis(metrics, diagnostic_checks)

    avg_delay_by_time = time_window_summary[
        ["time_label", "avg_delay", "record_count", "otp_percent", "outlier_count"]
    ].copy() if not time_window_summary.empty else pd.DataFrame()
    if not avg_delay_by_time.empty and avg_delay_by_time["avg_delay"].notna().any():
        spike_threshold = avg_delay_by_time["avg_delay"].mean() + avg_delay_by_time["avg_delay"].std()
        avg_delay_by_time["is_spike"] = avg_delay_by_time["avg_delay"] > spike_threshold

    return {
        "file_info": {
            "file_name": file_name,
            "file_type": Path(file_name).suffix.lstrip(".").lower() or "unknown",
            "row_count": int(len(working)),
            "column_names": list(working.columns),
        },
        "file_understanding": {
            "data_grain": grain,
            "available_dimensions": {key: value for key, value in dimensions.items() if value},
            "has_time_context": parsed_time is not None,
        },
        "quality_checks": quality,
        "performance_metrics": metrics,
        "segmentation_analysis": {
            "by_trip": to_serializable_records(trip_summary, limit=25),
            "by_station": to_serializable_records(station_summary, limit=25),
            "by_hour": to_serializable_records(hour_summary),
            "by_direction": to_serializable_records(direction_summary),
        },
        "diagnostic_checks": diagnostic_checks,
        "chart_ready_outputs": {
            "delay_distribution": compute_delay_distribution(delay_series),
            "delay_jump_events": to_serializable_records(delay_jump_events, limit=15),
            "focus_trip_profiles": to_serializable_records(focus_trip_profiles, limit=80),
            "avg_delay_by_trip": to_serializable_records(
                trip_summary[
                    ["group_value", "avg_delay", "record_count", "otp_percent", "outlier_count", "missing_stops"]
                ]
                .rename(columns={"group_value": "trip_id"})
                .sort_values("avg_delay", ascending=False),
                limit=15,
            )
            if not trip_summary.empty
            else [],
            "avg_delay_by_station": to_serializable_records(
                station_summary[
                    ["group_value", "avg_delay", "record_count", "otp_percent", "outlier_count"]
                ]
                .rename(columns={"group_value": "stop_id"})
                .sort_values("avg_delay", ascending=False),
                limit=15,
            )
            if not station_summary.empty
            else [],
            "avg_delay_by_hour": to_serializable_records(avg_delay_by_time),
            "station_hour_heatmap": to_serializable_records(
                station_hour_summary.sort_values(["stop_id", "hour"])
            )
            if not station_hour_summary.empty
            else [],
            "otp_by_trip": to_serializable_records(
                trip_summary[["group_value", "otp_percent", "record_count", "avg_delay", "outlier_count"]]
                .rename(columns={"group_value": "trip_id"})
                .sort_values("otp_percent"),
                limit=15,
            )
            if not trip_summary.empty
            else [],
            "otp_by_station": to_serializable_records(
                station_summary[["group_value", "otp_percent", "record_count", "avg_delay", "outlier_count"]]
                .rename(columns={"group_value": "stop_id"})
                .sort_values("otp_percent"),
                limit=15,
            )
            if not station_summary.empty
            else [],
            "otp_by_hour": to_serializable_records(hour_summary[["hour", "otp_percent"]])
            if not hour_summary.empty
            else [],
            "match_status_breakdown": to_serializable_records(match_status_summary),
            "missing_stop_sequence_summary": to_serializable_records(stop_gap_summary, limit=20),
        },
        "evidence_summary": evidence_summary,
    }
