from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


REQUIRED_PROCESSED_COLUMNS = {
    "stationId",
    "destination_stationId",
    "hour",
    "lat",
    "lng",
    "destination_lat",
    "destination_lng",
    "delta_time",
    "timestamp",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and attach the config path for relative path resolution."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    """Resolve a path relative to the config directory."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config["_config_dir"]) / path).resolve()


def _fallback_existing_path(config: dict[str, Any], candidates: list[str]) -> Path:
    for candidate in candidates:
        path = resolve_path(config, candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these input files exist: {candidates}")


def get_processed_data_path(config: dict[str, Any]) -> Path:
    configured = config.get("paths", {}).get("processed_data", "../../data/processed_data.csv")
    return _fallback_existing_path(
        config,
        [
            configured,
            "../../data/processed_data.csv",
            "../../data/proccessed_data.csv",
            "data/processed_data.csv",
            "data/proccessed_data.csv",
        ],
    )


def get_output_log_path(config: dict[str, Any]) -> Path:
    configured = config.get("paths", {}).get("output_log", "../../data/output_log.csv")
    return _fallback_existing_path(
        config,
        [
            configured,
            "../../data/output_log.csv",
            "../../data/ouput_log.csv",
            "data/output_log.csv",
            "data/ouput_log.csv",
        ],
    )


def parse_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp safely and backfill from date/time when needed."""
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif {"date", "time"}.issubset(out.columns):
        out["timestamp"] = pd.to_datetime(
            out["date"].astype(str) + " " + out["time"].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError("Missing timestamp column and cannot reconstruct from date/time.")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    else:
        out["date"] = out["timestamp"].dt.date.astype("string")

    if "time" not in out.columns:
        out["time"] = out["timestamp"].dt.strftime("%H:%M:%S")

    if "hour" not in out.columns:
        out["hour"] = out["timestamp"].dt.hour
    else:
        out["hour"] = pd.to_numeric(out["hour"], errors="coerce")

    return out


def load_processed_data(config: dict[str, Any]) -> pd.DataFrame:
    path = get_processed_data_path(config)
    df = pd.read_csv(path)
    missing = REQUIRED_PROCESSED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Processed data missing required columns: {sorted(missing)}")
    df = parse_timestamp_columns(df)
    LOGGER.info("Loaded processed data from %s with %d rows.", path, len(df))
    return df


def load_output_log(config: dict[str, Any]) -> pd.DataFrame:
    path = get_output_log_path(config)
    df = pd.read_csv(path)
    if "estimate_time" not in df.columns:
        raise ValueError(f"Output log must contain estimate_time: {path}")
    if "timestamp" in df.columns:
        df = parse_timestamp_columns(df)
    LOGGER.info("Loaded output log from %s with %d rows.", path, len(df))
    return df


def merge_baseline_eta(processed: pd.DataFrame, output_log: pd.DataFrame) -> pd.DataFrame:
    """Merge Vietmap estimate_time into processed data using timestamp-aware keys."""
    df = processed.copy().reset_index(drop=True)
    log_df = output_log.copy().reset_index(drop=True)

    if "baseline_eta_secs" in df.columns:
        df = df.drop(columns=["baseline_eta_secs"])
    if "estimate_time" in df.columns:
        df = df.drop(columns=["estimate_time"])

    merge_keys = ["stationId", "destination_stationId", "timestamp"]
    if all(col in log_df.columns for col in merge_keys):
        cols = merge_keys + ["estimate_time"]
        baseline = log_df[cols].drop_duplicates(merge_keys, keep="first")
        merged = df.merge(baseline, on=merge_keys, how="left")
        if merged["estimate_time"].isna().any() and len(log_df) == len(df):
            LOGGER.warning("Timestamp merge left missing estimates; filling remaining rows by index.")
            merged["estimate_time"] = merged["estimate_time"].fillna(log_df["estimate_time"])
        df = merged
    elif len(log_df) == len(df):
        LOGGER.warning("Output log has no timestamp; using row-order merge for estimate_time.")
        df["estimate_time"] = log_df["estimate_time"]
    else:
        raise ValueError(
            "Cannot merge output log: provide timestamp keys or matching row count. "
            f"processed={len(df)}, output_log={len(log_df)}"
        )

    df["baseline_eta_secs"] = pd.to_numeric(df["estimate_time"], errors="coerce")
    df["actual_eta_secs"] = pd.to_numeric(df["delta_time"], errors="coerce")
    df["residual_secs"] = df["actual_eta_secs"] - df["baseline_eta_secs"]
    return df


def select_fixed_od_pair(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter one OD pair from config or select the most frequent pair when config is null."""
    fixed = config.get("fixed_trip", {})
    station_id = fixed.get("stationId")
    destination_station_id = fixed.get("destination_stationId")

    if station_id is None or destination_station_id is None:
        counts = (
            df.groupby(["stationId", "destination_stationId"])
            .size()
            .sort_values(ascending=False)
        )
        if counts.empty:
            raise ValueError("No OD pairs available.")
        station_id, destination_station_id = counts.index[0]
        LOGGER.warning(
            "fixed_trip is null; selected most frequent OD pair stationId=%s destination_stationId=%s (%d rows).",
            station_id,
            destination_station_id,
            int(counts.iloc[0]),
        )

    filtered = df[
        (df["stationId"].astype(str) == str(station_id))
        & (df["destination_stationId"].astype(str) == str(destination_station_id))
    ].copy()
    if filtered.empty:
        raise ValueError(f"No rows for fixed OD pair {station_id}->{destination_station_id}.")

    filtered = filtered.sort_values("timestamp").reset_index(drop=True)
    metadata = {
        "stationId": str(station_id),
        "destination_stationId": str(destination_station_id),
        "row_count": int(len(filtered)),
    }
    return filtered, metadata


def load_experiment_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load processed data, merge baseline ETA, sort, and filter fixed OD pair."""
    processed = load_processed_data(config)
    output_log = load_output_log(config)
    df = merge_baseline_eta(processed, output_log)
    df, metadata = select_fixed_od_pair(df, config)
    return df, metadata
