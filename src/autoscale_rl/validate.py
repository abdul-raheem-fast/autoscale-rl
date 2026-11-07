"""Stage 3 — validate the extracted trace BEFORE anything downstream trains.

Checks (schema, nulls, ranges, continuity, coverage, split):
  S1  file loads & has the expected columns/dtypes
  S2  no nulls anywhere
  R1  requests >= 0 and <= configured sanity cap
  R2  timestamps strictly increasing, no duplicates
  C1  minute continuity: fraction of missing minutes within a day-coverage
      tolerance (real traces have small gaps; floods of holes are bad data)
  C2  enough days for train/eval split (validate.min_days)
  C3  rate distribution sane (p99 within bound)

The report is written to reports/data_validation.json and the stage exits
non-zero on any failure so `dvc repro` stops BEFORE training on bad data.
"""
from __future__ import annotations

import sys

import pandas as pd

from .config import load_params, out_path, set_seeds, save_json
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                            log_summary_metrics, start_run)


def build_checks(df: pd.DataFrame, cfg: dict) -> list:
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    expected_cols = ["timestamp", "requests", "minute_of_day", "day_index"]
    numeric_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in expected_cols[1:]) if len(df.columns) else False
    ok = not df.empty and all(c in df.columns for c in expected_cols) and numeric_ok
    record("S1.schema", ok, f"columns={sorted(df.columns.tolist())}, dtypes={dict(df.dtypes.astype(str))}")
    if not ok:
        return checks

    record("S2.no-nulls", not df.isna().any().any(),
           f"null cells={int(df.isna().sum().sum())}")

    req = df["requests"]
    cap = int(cfg["max_rate_per_minute"])
    record("R1.rate-range", bool((req >= 0).all() and (req <= cap).all()),
           f"min={req.min()} max={req.max()} cap={cap}")
    dupe = df["timestamp"].duplicated().any()
    mono = df["timestamp"].is_monotonic_increasing
    record("R2.timestamps-monotonic-unique", bool(mono and not dupe),
           f"duplicates={int(dupe)} monotonic={bool(mono)}")

    first_day = df["timestamp"].min().normalize()
    if df["timestamp"].max().normalize() > first_day:
        span = (df["timestamp"].max().normalize() - first_day).days + 1
        expected = span * 1440
        missing_frac = 1.0 - len(df) / expected
    else:
        span = 1
        missing_frac = 0.0
    record("C1.minute-continuity", missing_frac <= float(cfg["max_missing_minutes_frac"]),
           f"missing_frac={missing_frac:.4f} (tolerance "
           f"{float(cfg['max_missing_minutes_frac']):.4f}, span_days={span})")

    n_days = int(df["day_index"].nunique())
    record("C2.enough-days", n_days >= int(cfg["min_days"]),
           f"days={n_days} min_days={int(cfg['min_days'])}")

    p99 = float(req.quantile(0.99))
    record("C3.rate-p99-sane", p99 <= cap, f"p99_rate={p99:.1f} cap={cap}")
    return checks


def train_eval_split(df: pd.DataFrame, train_frac: float) -> tuple:
    """First `train_frac` of days -> train; the rest -> eval (ordered split,
    so evaluation is always on calendar days the agent never saw)."""
    days = sorted(df["day_index"].unique())
    n_train = max(1, int(round(len(days) * train_frac)))
    return days[:n_train], days[n_train:]


def main() -> None:
    params = load_params()
    set_seeds(int(params["seed"]))
    cfg = params["validate"]
    df = pd.read_csv(out_path(params, "processed"), parse_dates=["timestamp"])

    # Coerce numerics WITHOUT dropping rows: a failed coercion becomes a NaN,
    # which the S2 (nulls) check then reports honestly.
    df["requests"] = pd.to_numeric(df["requests"], errors="coerce")
    df["minute_of_day"] = pd.to_numeric(df["minute_of_day"], errors="coerce")
    df["day_index"] = pd.to_numeric(df["day_index"], errors="coerce")
    df["requests"] = df["requests"].astype("float64")

    checks = build_checks(df, cfg)
    train_days, eval_days = train_eval_split(df, float(cfg["train_frac"]))
    passed = all(c["passed"] for c in checks) and len(eval_days) >= 3

    report = {
        "passed": bool(passed),
        "checks": checks,
        "summary": {
            "rows": int(len(df)),
            "days": int(df["day_index"].nunique()),
            "total_requests": int(df["requests"].sum()),
            "mean_rate_per_minute": float(df["requests"].mean()),
            "p99_rate_per_minute": float(df["requests"].quantile(0.99)),
            "max_rate_per_minute": int(df["requests"].max()),
            "date_start": str(df["timestamp"].min()),
            "date_end": str(df["timestamp"].max()),
        },
        "split": {"train_days": [int(d) for d in train_days],
                  "eval_days": [int(d) for d in eval_days]},
    }
    report_path = out_path(params, "validation_report")
    save_json(report_path, report)
    if not passed:
        print(f"[validate] FAILED: {[c['name'] for c in checks if not c['passed']]}")
        print(f"[validate] full report: {report_path}")

    mlflow, _ = configure_mlflow(params)
    with start_run(mlflow, run_name="data-validation") as run:
        log_params_flat(mlflow, "validate.", cfg)
        log_summary_metrics(mlflow, {"validation_passed": int(passed),
                                     **{f"check_{c['name'].split('.')[0]}_{i}": int(c["passed"])
                                        for i, c in enumerate(checks)}})
        log_artifact(mlflow, str(report_path), artifact_dir="data/validation")
    print(f"[validate] MLflow run: {run.info.run_id}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
