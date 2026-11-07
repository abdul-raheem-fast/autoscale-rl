"""Stage 1 — ingest public web-trace data from its source.

Downloads the NASA HTTP server access trace (July 1995), a classic dataset
from the Internet Traffic Archive, replayed in the simulator ("replayed
public web traces"). The primary URL is an immutable Wayback Machine snapshot
so the blob is byte-identical on every machine.

Candidate URLs are tried in order (see params.yaml). If downloading is
impossible (e.g. fully offline machine) and `allow_synthetic_fallback` is
true, a synthetic log in the SAME format is generated with a fixed seed and
the run is honestly labelled `source=synthetic-fallback` in
data/raw/trace_meta.json so downstream users never mistake it for real data.

Outputs (DVC-tracked):
    data/raw/trace.log        plain-text access log
    data/raw/trace_meta.json  sha256, byte count, line count, source, url
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import random
import urllib.request
from pathlib import Path

from .config import load_params, out_path, set_seeds, save_json, smoke
from .mlflow_utils import (configure_mlflow, log_artifact, log_params_flat,
                           log_summary_metrics, start_run)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_trace(urls: list, dst_gz: Path, timeout: int) -> tuple:
    """Try each URL; return (url_that_worked, downloaded_bytes)."""
    for url in urls:
        try:
            print(f"[ingest] Trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "autoscale-rl/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 1000 or data[:2] != b"\x1f\x8b":
                print(f"[ingest] {url}: not a gzip payload, skipping")
                continue
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:   # probe decompression
                if not gz.read(4096):
                    continue
            dst_gz.write_bytes(data)
            print(f"[ingest] OK: {len(data) / 1e6:.1f} MB from {url}")
            return url, len(data)
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"[ingest] {url} failed: {exc}")
    raise RuntimeError("all configured trace URLs failed")


def write_synthetic_log(path: Path, params: dict, seed: int) -> None:
    """Offline fallback: generate the same access-log format with a fixed seed.

    Traffic profile: two diurnal humps, weekday/weekend modulation and three
    flash-crowd bursts — a plausible replica of real web-service traffic.
    """
    import numpy as np

    cfg = params["ingest"]
    start = dt.datetime.strptime(cfg["synthetic_start"], "%Y-%m-%d")
    days = int(cfg["synthetic_days"])
    rng = np.random.default_rng(seed)
    random.seed(seed)

    paths = ["/index.html", "/news.html", "/images/logo.gif", "/history/apollo/",
             "/history/apollo/nasa.gov.html", "/shuttle/countdown/", "/search?q=atlantis"]
    ips = [f"199.72.81.{i}" for i in range(1, 60)] + \
          [f"205.212.{i % 256}.{(i * 7) % 256}" for i in range(1, 60)]

    burst_days = {3: 4.0, 11: 8.0, 19: 5.0}   # flash-crowd multipliers
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        minute = dt.timedelta(minutes=1)
        for day in range(days):
            t0 = start + dt.timedelta(days=day)
            weekend = t0.weekday() >= 5
            burst_mult = burst_days.get(day, 1.0)
            for hrs in range(24):
                hour = t0 + dt.timedelta(hours=hrs)
                weekday_hump = 25.0 * float(np.exp(-(((hrs - 13) ** 2) / 8.0)))
                evening_hump = 14.0 * float(np.exp(-(((hrs - 19) ** 2) / 12.0)))
                night_floor = 6.0
                rate = night_floor + weekday_hump + evening_hump
                rate *= 0.55 if weekend else 1.0
                rate *= burst_mult
                for mins in range(60):
                    arrivals = int(rng.poisson(rate))
                    ts0 = hour + dt.timedelta(minutes=mins)
                    for _ in range(arrivals):
                        ts = ts0 + dt.timedelta(seconds=random.randrange(60))
                        ip = random.choice(ips)
                        path_sel = random.choice(paths)
                        nbytes = random.choice(("-", "300", "6246", "1483", "4085"))
                        f.write(
                            f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S")} -0400] '
                            f'"GET {path_sel} HTTP/1.0" 200 {nbytes}\n'
                        )


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    params = load_params()
    seed = int(params["seed"])
    set_seeds(seed)
    raw = out_path(params, "raw_trace")
    meta_path = out_path(params, "raw_meta")
    raw.parent.mkdir(parents=True, exist_ok=True)
    cfg = params["ingest"]
    tmp_gz = raw.parent / "trace_download.gz.tmp"
    try:
        tmp_gz.unlink(missing_ok=True)
    except FileNotFoundError:
        pass

    source, url, sha = "synthetic-fallback", None, None
    try:
        url, nbytes = download_trace(cfg["trace_urls"], tmp_gz, int(cfg["timeout_seconds"]))
        with open(tmp_gz, "rb") as binf:
            sha = _sha256_bytes(binf.read())
        with gzip.open(tmp_gz, "rb") as gz, open(raw, "wb") as out:
            out.write(gz.read())
        source = f"download:{cfg['trace_name']}"
    except Exception as exc:
        if not cfg.get("allow_synthetic_fallback", False):
            raise
        print(f"[ingest] Download failed ({exc}); writing SYNTHETIC fallback trace "
              "(meta records this honestly).")
        write_synthetic_log(raw, params, seed)

    n_lines = count_lines(raw)
    sha = sha or _sha256_file(raw)
    meta = {
        "name": cfg["trace_name"],
        "source": source,
        "url": url,
        "sha256": sha,
        "bytes": raw.stat().st_size,
        "lines": n_lines,
        "script": "src/autoscale_rl/ingest.py",
    }
    save_json(meta_path, meta)
    print(f"[ingest] source={source} lines={n_lines} sha256={sha[:12]}...")
    try:
        tmp_gz.unlink()
    except FileNotFoundError:
        pass

    mlflow, _ = configure_mlflow(params)
    with start_run(mlflow, run_name="data-ingest") as run:
        log_params_flat(mlflow, "ingest.", {
            "trace.name": cfg["trace_name"],
            "seed": seed,
            "n_urls_configured": len(cfg["trace_urls"]),
            "allow_synthetic_fallback": cfg.get("allow_synthetic_fallback", False),
        })
        log_summary_metrics(mlflow, {"raw_bytes": meta["bytes"], "raw_lines": n_lines})
        log_artifact(mlflow, str(meta_path), artifact_dir="data/manifests")
    print(f"[ingest] MLflow run: {run.info.run_id if run else '(none)'}")


if __name__ == "__main__":
    main()
