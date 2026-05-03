import glob

import pytest
import yaml

CONFIGS = sorted(glob.glob("configs/*.yaml"))
ENCODER_CONFIGS = [c for c in CONFIGS if "encoder" in c]
HEAD_CONFIGS = [c for c in CONFIGS if "head" in c]


@pytest.mark.parametrize("path", CONFIGS)
def test_config_yaml_loads(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("path", ENCODER_CONFIGS)
def test_encoder_config_buildable(path):
    from jepa.model import build_model

    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    assert model is not None
    assert "vicreg_target" in cfg["training"], (
        f"{path} missing training.vicreg_target — required since the gradient-flowing "
        "VICReg switch was added"
    )
    assert cfg["training"]["vicreg_target"] in {"predicted", "target"}


@pytest.mark.parametrize("path", HEAD_CONFIGS)
def test_head_config_required_keys(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("embeddings_file", "playlists_file", "task"):
        assert k in cfg["data"], f"{path} missing data.{k}"
    for k in ("hidden_dim", "residual_source"):
        assert k in cfg["model"], f"{path} missing model.{k}"
    assert cfg["data"]["task"] in {"continuation", "infill"}


@pytest.mark.parametrize("path", HEAD_CONFIGS)
def test_head_out_path_matches_dataset(path):
    """Sample variants must write into checkpoints-sample/, full into checkpoints/.
    Caught a real bug where the sample configs silently overwrote full-run heads.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = cfg["training"]["out"]
    if "_sample" in path:
        assert out.startswith("checkpoints-sample/"), f"{path} writes to {out}"
    else:
        assert out.startswith("checkpoints/") and "checkpoints-sample" not in out, (
            f"{path} writes to {out}"
        )
