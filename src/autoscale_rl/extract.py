"""Stage 2 — extract: raw HTTP access log -> per-minute request counts.

Parses the NASA-format common log format line by line (fast manual parse of
the timestamp; tolerant of malformed lines, which get counted and skipped)
and aggregates requests into one row per minute.

Output (DVC-tracked):
    data/processed/requests.csv
      timestamp          minute bucket start (naive server-local time)
      requests           number of logged requests in that minute
      minute_of_day      0..1439
      day_index          calendar day number starting at 0
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from .config import load_params, out_path, set_seeds
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                           log_summary_metrics, start_run)

MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

VALID_METHODS = {"GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS", "PROPFIND"}


def parse_ts(buf: str) -> tuple:   # "01/Jul/1995:00:00:01 -0400" → (year, month, day, minute)
    try:
        year = int(buf[7:11])
        month = MONTHS[buf[3:6]]
        day = int(buf[0:2])
        hour = int(buf[12:14])
        minute = int(buf[15:17])
        return year, month, day, hour, minute
    except (ValueError, KeyError):
        raise


def parse_line(line: str):
    """Return a (minute-key, is_request) tuple or None for malformed lines."""
    sq = line.find("[")
    eq = line.find("]")
    if sq == -1 or eq == -1 or eq < sq:
        return None
    try:
        key = parse_ts(line[sq + 1:eq])
    except Exception:
        return None
    q1 = line.find('"')
    q2 = line.find('"', q1 + 1)
    if q1 == -1 or q2 == -1:
        return None
    parts = line[q1 + 1:q2].split()
    if len(parts) < 2 or parts[0] not in VALID_METHODS:
        return None
    return key


def extract(raw_log_path, out_csv_path) -> dict:
    counts = Counter()
    seen_dates = set()
    malformed = 0
    with open(raw_log_path, "r", encoding="latin-1", newline="") as f:
        for line in f:
            key = parse_line(line)
            if key is None:
                malformed += 1
                continue
            year, month, day, hour, minute = key
            counts[key] += 1
            seen_dates.add((year, month, day))

    if not counts:
        raise RuntimeError("no parseable lines in trace; ingestion suspect")

    rows = []
    for (year, month, day, hour, minute) in sorted(counts.keys()):
        ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        rows.append({
            "timestamp": ts,
            "requests": counts[(year, month, day, hour, minute)],
            "minute_of_day": hour * 60 + minute,
        })
    df = pd.DataFrame(rows)
    first_day = pd.Timestamp(year=sorted(seen_dates)[0][0],
                             month=sorted(seen_dates)[0][1],
                             day=sorted(seen_dates)[0][2])
    df["day_index"] = (df["timestamp"].dt.normalize() - first_day).dt.days
    df.to_csv(out_csv_path, index=False)

    return {
        "rows": int(len(df)),
        "total_requests": int(df["requests"].sum()),
        "minutes_covered": int(df["timestamp"].nunique()),
        "days": int(df["day_index"].nunique()),
        "malformed_lines": malformed,
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
        "mean_rate_per_minute": float(df["requests"].mean()),
        "p99_rate_per_minute": float(df["requests"].quantile(0.99)),
    }


def main() -> None:
    params = load_params()
    set_seeds(int(params["seed"]))
    raw = out_path(params, "raw_trace")
    out_csv = out_path(params, "processed")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    stats = extract(raw, out_csv)
    print(f"[extract] {stats['total_requests']:,} requests -> "
          f"{stats['rows']:,} minute rows over {stats['days']} days "
          f"({stats['malformed_lines']:,} malformed lines skipped)")

    mlflow, _ = configure_mlflow(params)
    with start_run(mlflow, run_name="trace-extract") as run:
        log_params_flat(mlflow, "extract.", {"log": str(raw.name)})
        log_summary_metrics(mlflow, stats)
        log_artifact(mlflow, str(out_csv), artifact_dir="data/extracts")
    print(f"[extract] MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
