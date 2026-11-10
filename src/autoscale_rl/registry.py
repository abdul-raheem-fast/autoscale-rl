"""MLflow Model Registry: register a trained checkpoint and move it to Staging.

Runs against whatever tracking server is configured (DagsHub for the
submission). When the server is unreachable / not configured, the function
fails softly: the pipeline still completes, and comparison.json records
`registration: "skipped (no remote tracking server)"`.
"""
from __future__ import annotations

import sys

from .mlflow_utils import is_remote_tracking


VALID_STAGES = {"Staging", "Production"}


def register_and_stage(model_uri: str, model_name: str, stage: str,
                       extra_tags: dict | None = None) -> str:
    """Register model under `model_name` and transition to `stage` (Staging).

    Returns a status string describing what happened.
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unsupported model stage {stage!r}; choose Staging or Production")

    import mlflow
    from mlflow.tracking import MlflowClient

    if not is_remote_tracking():
        note = "skipped (MLFLOW_TRACKING_URI not set — no remote tracking server)"
        print(f"[registry] {note}", file=sys.stderr)
        return note

    # --- registration (DagsHub-hosted registry) ---
    model_version = mlflow.register_model(model_uri, name=model_name)
    version = model_version.version
    print(f"[registry] registered {model_name} v{version} from {model_uri}")

    client = MlflowClient()
    if extra_tags:
        for k, v in extra_tags.items():
            client.set_model_version_tag(model_name, version, k, str(v))

    # staged status (MLflow 2.x API, which DagsHub serves)
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=False)
    print(f"[registry] {model_name} v{version} -> {stage}")

    try:  # describe registered model briefly
        details = client.get_model_version(model_name, version)
        return f"registered {model_name} v{version} run={details.run_id} stage={stage}"
    except (AttributeError, RuntimeError):
        return f"registered {model_name} v{version} -> {stage}"
