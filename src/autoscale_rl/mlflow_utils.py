"""MLflow plumbing: tracking server resolution, run/param/artifact logging.

Tracking server
---------------
For the milestone, tracking + registry live on DagsHub (NOT local files):

    MLFLOW_TRACKING_URI       https://dagshub.com/<owner>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME  <dagshub username>
    MLFLOW_TRACKING_PASSWORD  <dagshub access token>

`get_tracking_uri` falls back to a local `mlruns/` file store only when the
tracking URI is not configured, so the pipeline still runs offline. The
fallback prints a prominent warning and tags those runs with
`tracking.server=fallback-local` so they are never mistaken for runs on the
hosted server.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional


def get_tracking_uri() -> str:
    """Resolve the tracking URI: env MLFLOW_TRACKING_URI > DagsHub convention > local fallback."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    user = os.environ.get("DAGSHUB_USER")
    # Repo owner may differ from the authenticating user (team collaborators).
    owner = os.environ.get("DAGSHUB_OWNER") or user
    repo = os.environ.get("DAGSHUB_REPO", "autoscale-rl")
    if owner and user:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        # DagsHub accepts the access token as password
        token = os.environ.get("DAGSHUB_TOKEN")
        if token:
            os.environ.setdefault("MLFLOW_TRACKING_USERNAME", user)
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
        return uri
    # --- local fallback (offline smoke test only) ---
    print(
        "[mlflow] WARNING: MLFLOW_TRACKING_URI is not set — logging to the LOCAL "
        "'mlruns/' file store. Configure DagsHub for real runs:\n"
        "  export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow\n"
        "  export MLFLOW_TRACKING_USERNAME=<user>\n"
        "  export MLFLOW_TRACKING_PASSWORD=<token>",
        file=sys.stderr,
    )
    return "local"


def is_remote_tracking() -> bool:
    uri = get_tracking_uri()
    return uri.startswith("http")


def configure_mlflow(params: Dict[str, Any]):
    """Import mlflow, set URI/experiment. Returns (mlflow, experiment_name)."""
    import mlflow

    tracking_uri = get_tracking_uri()
    if tracking_uri == "local":
        mlflow.set_tracking_uri(None)  # file store, ./mlruns
    else:
        mlflow.set_tracking_uri(tracking_uri)
    name = params["mlflow"]["experiment_name"]
    mlflow.set_experiment(name)
    return mlflow, name


def start_run(mlflow, run_name: str, tags: Optional[Dict[str, str]] = None):
    extra: Dict[str, str] = {"mlflow.runName": run_name}
    if not is_remote_tracking():
        extra["tracking.server"] = "fallback-local"
    if tags:
        extra.update(tags)
    return mlflow.start_run(run_name=run_name, tags=extra)


def log_params_flat(mlflow, prefix: str, data: Dict[str, Any]) -> None:
    """Log a dict as MLflow params with a `prefix/key` naming scheme.

    MLflow params are depth-1, so nested configs are flattened and stringified;
    long values are truncated to the 500-char param limit.
    """
    def flatten(obj, parent=""):
        items: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                items.update(flatten(v, f"{parent}{k}."))
        elif isinstance(obj, list):
            items[parent[:-1]] = "[" + ", ".join(map(str, obj)) + "]"
        else:
            items[parent[:-1]] = obj
        return items

    flat = flatten(data)
    for k, v in flat.items():
        s = "None" if v is None else str(v)
        if len(s) > 500:
            s = s[:497] + "..."
        mlflow.log_param(f"{prefix}{k}", s)


def log_summary_metrics(mlflow, metrics: Dict[str, Any]) -> None:
    import math
    clean = {}
    for k, v in metrics.items():
        if v is None or isinstance(v, (str, bool)):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            fv = 0.0
        clean[k] = fv
    mlflow.log_metrics(clean)


def log_int_metric(mlflow, key: str, step: int, value: float) -> None:
    """Per-step/episode metrics (reward curves etc.)."""
    if value is None:
        return
    value = float(value)
    if value != value or abs(value) == float("inf"):
        value = 0.0
    mlflow.log_metric(key, value, step=step)


def log_artifact(mlflow, path: str, artifact_dir: Optional[str] = None) -> None:
    try:
        if artifact_dir:
            mlflow.log_artifact(path, artifact_path=artifact_dir)
        else:
            mlflow.log_artifact(path)
    except Exception as exc:  # never fail a pipeline stage on artifact upload
        print(f"[mlflow] artifact upload failed for {path}: {exc}", file=sys.stderr)
