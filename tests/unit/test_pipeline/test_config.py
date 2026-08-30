"""
tests/unit/test_pipeline/test_config.py
---------------------------------------
Unit tests for PipelineConfig, ModelSpec, and DatasetSpec (Milestone 4).
"""

import json
import pytest
from pathlib import Path
from nemo_eval.pipeline.config import PipelineConfig, ModelSpec, DatasetSpec, ExecutionMode


class TestPipelineConfig:
    def test_default_pipeline_config(self):
        config = PipelineConfig()
        assert config.run_label == "metacognition_eval_run"
        assert config.mode == ExecutionMode.BOTH
        assert config.max_turns == 25
        assert config.enable_telemetry is True

    def test_model_spec_validation(self):
        spec = ModelSpec(
            name="Qwen2.5-Math-7B",
            provider="mock",
            model_id="qwen-math-7b"
        )
        assert spec.name == "Qwen2.5-Math-7B"
        assert spec.provider == "mock"

    def test_dataset_spec_validation(self):
        spec = DatasetSpec(
            name="math",
            max_tasks=25,
            split="test"
        )
        assert spec.name == "math"
        assert spec.max_tasks == 25

    def test_json_roundtrip(self, tmp_path):
        config_path = tmp_path / "test_config.json"
        config = PipelineConfig(
            run_label="test_sweep",
            output_dir=str(tmp_path / "results"),
            mode="vanilla",
            models=[ModelSpec(name="Phi4-mini-reasoning", provider="mock")],
            datasets=[DatasetSpec(name="lila", max_tasks=10)]
        )
        config.to_json(config_path)
        assert config_path.exists()

        loaded = PipelineConfig.from_json(config_path)
        assert loaded.run_label == "test_sweep"
        assert loaded.mode == "vanilla"
        assert len(loaded.models) == 1
        assert loaded.models[0].name == "Phi4-mini-reasoning"
        assert len(loaded.datasets) == 1
        assert loaded.datasets[0].max_tasks == 10
