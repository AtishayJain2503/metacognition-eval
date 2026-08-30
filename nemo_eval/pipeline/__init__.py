"""
nemo_eval.pipeline
------------------
Evaluation Pipeline: runner, config, and reporter (Milestone 4).

Exports:
    - PipelineConfig: YAML/JSON-loadable evaluation configuration.
    - ModelSpec: Model specification container.
    - DatasetSpec: Dataset specification container.
    - ExecutionMode: Execution mode enum (vanilla, agentic, both, dual_parity).
    - BenchmarkRunner: Multi-dataset, multi-model evaluation sweep engine.
    - RunRecord: Aggregated results container for an evaluation slice.
    - MarkdownReporter / PipelineReporter: Scorecard, comparison table, and leaderboard generator.
"""

from nemo_eval.pipeline.config import (
    DatasetSpec,
    ExecutionMode,
    ModelSpec,
    PipelineConfig,
)
from nemo_eval.pipeline.reporter import (
    MarkdownReporter,
    PipelineReporter,
)
from nemo_eval.pipeline.runner import (
    BenchmarkRunner,
    RunRecord,
)

__all__ = [
    "PipelineConfig",
    "ModelSpec",
    "DatasetSpec",
    "ExecutionMode",
    "BenchmarkRunner",
    "RunRecord",
    "MarkdownReporter",
    "PipelineReporter",
]
