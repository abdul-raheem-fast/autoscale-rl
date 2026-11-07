"""Shared configuration, seeding and small IO helpers.

Every pipeline stage loads `params.yaml` through :func:`load_params` so that
DVC can hash the section it depends on, and seeds all RNG sources through
:func:`set_seeds` so runs are bit-for-bit reproducible.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]          # .../autoscale-rl
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"


def _repo_path(p: str | Path) -> Path:
    """Resolve a relative path from params.yaml against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_params(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load params.yaml (or the path given via --params / PARAMS env var)."""
    declared = os.environ.get("PARAMS")
    params_path = _repo_path(path or declared or DEFAULT_PARAMS)
    with open(params_path, "r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f)


def out_path(params: Dict[str, Any], key: str) -> Path:
    """Resolve a path declared under `paths:` in params.yaml."""
    return _repo_path(params["paths"][key])


def smoke() -> bool:
    """Smoke-testing switch: shrinks train/eval loops so the whole pipeline
    finishes in seconds. Selected by AUTOSCALE_SMOKE=1 — never in submission runs."""
    return os.environ.get("AUTOSCALE_SMOKE", "") in {"1", "true", "True"}


def set_seeds(seed: int) -> None:
    """Seed python, numpy and torch so every stage is deterministic."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # stages that don't need torch
        pass


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
