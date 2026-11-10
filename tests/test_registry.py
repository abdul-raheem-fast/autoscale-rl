import pytest

from autoscale_rl.registry import register_and_stage


def test_registry_rejects_unknown_stage_before_importing_mlflow():
    with pytest.raises(ValueError, match="choose Staging or Production"):
        register_and_stage("runs:/example/model", "AutoScaleRLPolicy", "Archived")
