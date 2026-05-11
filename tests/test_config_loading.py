"""Test YAML config loading and FBDConfig construction."""

import pytest
from pathlib import Path
from omegaconf import OmegaConf

from fbd_lora.config import FBDConfig


class TestConfigLoading:
    """YAML configs must load and produce valid FBDConfig instances."""

    def test_fbd_config_defaults(self):
        """FBDConfig should have correct default values."""
        cfg = FBDConfig()
        assert cfg.enabled is True
        assert cfg.routing_type == "pullback_metric_mixed"
        assert cfg.metric_mode == "diag"
        assert cfg.route_a is True
        assert cfg.route_b is False
        assert cfg.lambda_route == 0.25
        assert cfg.epsilon == 1e-4
        assert cfg.norm_match is True
        assert cfg.alignment_gate is True
        assert cfg.gate_type == "hard"
        assert cfg.metric_cache is True
        assert cfg.log_gradient_stats is True
        assert cfg.gradient_stats_interval == 10

    def test_fbd_config_from_dict(self):
        """FBDConfig.from_dict must handle extra keys gracefully."""
        d = {
            "enabled": True,
            "routing_type": "pullback_metric_mixed",
            "lambda_route": 0.5,
            "unknown_key": "ignored",  # extra keys should be filtered
        }
        cfg = FBDConfig.from_dict(d)
        assert cfg.lambda_route == 0.5
        assert cfg.routing_type == "pullback_metric_mixed"

    def test_load_metamath_fbd_yaml(self):
        """Main FBD config YAML must load successfully."""
        yaml_path = Path("configs/nlg/metamath_fbd.yaml")
        if not yaml_path.exists():
            pytest.skip(f"Config not found: {yaml_path}")

        cfg = OmegaConf.load(yaml_path)
        assert OmegaConf.select(cfg, "modality") == "nlg"
        assert OmegaConf.select(cfg, "task") == "metamath"
        assert OmegaConf.select(cfg, "fbd.enabled") is True
        assert OmegaConf.select(cfg, "adapter.rank") == 16
        assert OmegaConf.select(cfg, "fbd.lambda_route") == 0.25

    def test_load_lora_baseline_yaml(self):
        """LoRA baseline config must have fbd.enabled=false."""
        yaml_path = Path("configs/nlg/baselines/lora.yaml")
        if not yaml_path.exists():
            pytest.skip(f"Config not found: {yaml_path}")

        cfg = OmegaConf.load(yaml_path)
        assert OmegaConf.select(cfg, "adapter.name") == "lora"
        fbd_enabled = OmegaConf.select(cfg, "fbd.enabled", default=False)
        assert str(fbd_enabled).lower() in ("false", "0", "no") or fbd_enabled is False

    def test_fbd_config_from_yaml_section(self):
        """FBD config section from YAML must parse into FBDConfig."""
        yaml_path = Path("configs/nlg/metamath_fbd.yaml")
        if not yaml_path.exists():
            pytest.skip(f"Config not found: {yaml_path}")

        cfg = OmegaConf.load(yaml_path)
        fbd_dict = OmegaConf.to_container(OmegaConf.select(cfg, "fbd"), resolve=True)
        fbd_config = FBDConfig.from_dict(fbd_dict)

        assert fbd_config.enabled is True
        assert fbd_config.lambda_route == 0.25
        assert fbd_config.metric_mode == "diag"
        assert fbd_config.routing_type == "pullback_metric_mixed"

    def test_config_immutability_of_dataclass(self):
        """FBDConfig fields must be accessible and type-safe."""
        cfg = FBDConfig(lambda_route=0.3, metric_mode="full")
        assert cfg.lambda_route == 0.3
        assert cfg.metric_mode == "full"
        # target_modules default is empty list
        assert isinstance(cfg.target_modules, list)
