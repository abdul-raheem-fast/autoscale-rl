import numpy as np
import pandas as pd
import pytest

from autoscale_rl.validate import build_checks, train_eval_split


def make_valid_df(days=4):
    ts = pd.date_range("1995-07-01", periods=days * 1440, freq="min")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": ts,
        "requests": rng.integers(10, 400, size=len(ts)),
        "minute_of_day": list(range(1440)) * days,
        "day_index": np.repeat(np.arange(days), 1440),
    }).astype({"requests": "int64", "minute_of_day": "int64", "day_index": "int64"})


def test_clean_trace_passes():
    cfg = {"max_rate_per_minute": 5000, "min_days": 3, "max_missing_minutes_frac": 0.05}
    checks = build_checks(make_valid_df(), cfg)
    assert all(c["passed"] for c in checks), checks


def test_nulls_fail():
    df = make_valid_df()
    df.loc[10, "requests"] = np.nan
    df = df.astype({"requests": "float64"})
    checks = build_checks(df, {"max_rate_per_minute": 5000, "min_days": 3,
                               "max_missing_minutes_frac": 0.05})
    assert any(not c["passed"] for c in checks)


def test_out_of_range_rates_fail():
    df = make_valid_df()
    df.loc[5, "requests"] = 999999   # > cap
    checks = build_checks(df, {"max_rate_per_minute": 5000, "min_days": 3,
                               "max_missing_minutes_frac": 0.05})
    failed = {c["name"] for c in checks if not c["passed"]}
    assert failed


def test_split_preserves_order_and_coverage():
    df = make_valid_df(days=10)
    train, ev = train_eval_split(df, 0.8)
    assert train == [0, 1, 2, 3, 4, 5, 6, 7]
    assert ev == [8, 9]
