"""
nemo_eval.pipeline
------------------
Evaluation Pipeline: runner, config, and reporter (Milestone 5).

Exports:
    - PipelineConfig: YAML/JSON-loadable evaluation configuration.
    - BenchmarkRunner: Multi-dataset, multi-model evaluation harness.
    - PipelineReporter: Scorecard and failure trace generator.
"""

from nemo_eval.pipeline.config import PipelineConfig, ModelSpec, DatasetSpec
from nemo_eval.pipeline.runner import BenchmarkRunner
from nemo_eval.pipeline.reporter import PipelineReporter

__all__ = [
    "PipelineConfig",
    "ModelSpec",
    "DatasetSpec",
    "BenchmarkRunner",
    "PipelineReporter",
]
