"""Tracking-URI resolution: owner vs. authenticating user on DagsHub."""
import pytest

from autoscale_rl.mlflow_utils import get_tracking_uri

ENV_KEYS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
            "DAGSHUB_USER", "DAGSHUB_OWNER", "DAGSHUB_REPO", "DAGSHUB_TOKEN"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_explicit_uri_wins(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/x.mlflow")
    monkeypatch.setenv("DAGSHUB_USER", "someone")
    assert get_tracking_uri() == "https://example.com/x.mlflow"


def test_owner_defaults_to_user(monkeypatch):
    monkeypatch.setenv("DAGSHUB_USER", "alice")
    assert get_tracking_uri() == "https://dagshub.com/alice/autoscale-rl.mlflow"


def test_collaborator_points_at_owner_repo_with_own_creds(monkeypatch):
    import os
    monkeypatch.setenv("DAGSHUB_USER", "bob")
    monkeypatch.setenv("DAGSHUB_OWNER", "alice")
    monkeypatch.setenv("DAGSHUB_TOKEN", "bob-token")
    assert get_tracking_uri() == "https://dagshub.com/alice/autoscale-rl.mlflow"
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == "bob"
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == "bob-token"


def test_local_fallback_without_config():
    assert get_tracking_uri() == "local"
