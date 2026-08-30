"""
nemo_eval.pipeline.config
--------------------------
Pipeline configuration models — loadable from YAML, JSON, or Python dicts.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class ExecutionMode(str, Enum):
    VANILLA = "vanilla"
    AGENTIC = "agentic"
    BOTH = "both"
    DUAL_PARITY = "dual_parity"


class ModelSpec(BaseModel):
    """Specification for a single model inference provider."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Human-readable model label, e.g. 'Qwen2.5-Math-7B' or 'llama-3.3-70b-groq'.")
    provider: Literal["groq", "openai", "openai_gateway", "ollama", "vllm", "nemo", "nim", "mock", "mock_runner"] = "mock"
    model_id: str = Field(default="", description="Provider-specific model ID, e.g. 'Qwen/Qwen2.5-Math-7B-Instruct'.")
    api_key_env: Optional[str] = Field(default=None, description="Environment variable name for API key.")
    base_url: Optional[str] = Field(default=None, description="Base URL override for OpenAI-compatible endpoints.")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional provider kwargs.")


class DatasetSpec(BaseModel):
    """Specification for a benchmark dataset to evaluate against."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        default="math",
        description="Dataset name: 'math', 'putnam', 'lila', 'all', 'synthetic', 'gsm8k', 'infiagent', 'bird_sql', 'databench'."
    )
    category: Optional[str] = Field(default=None, description="Subcategory filter (e.g. 'algebra', 'calculus').")
    subdiscipline: Optional[str] = Field(default=None, description="Subdiscipline filter alias.")
    subject: Optional[str] = Field(default=None, description="Subject filter for MATH loader.")
    max_tasks: Optional[int] = Field(default=50, ge=1, description="Limit tasks loaded (useful for smoke tests).")
    data_dir: Optional[str] = Field(default=None, description="Path to dataset files (optional if using local fixtures).")
    split: str = "test"
    extra: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Top-level evaluation pipeline configuration."""
    model_config = ConfigDict(extra="ignore")

    run_label: str = Field(default="metacognition_eval_run", description="Human-readable run identifier.")
    output_dir: str = Field(default="./results", description="Directory for all run artifacts.")
    mode: Union[ExecutionMode, str] = Field(default=ExecutionMode.BOTH, description="Execution mode: vanilla, agentic, both / dual_parity.")
    models: List[ModelSpec] = Field(default_factory=list)
    datasets: List[DatasetSpec] = Field(default_factory=list)

    max_turns: int = Field(default=25, ge=1)
    max_correction_attempts: int = Field(default=3, ge=0)
    enable_planning: bool = Field(default=True)
    verify_intermediate: bool = Field(default=True)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)

    enable_telemetry: bool = Field(default=True)
    telemetry_sample_interval_ms: int = Field(default=20)
    num_samples: int = Field(default=1, ge=1, description="Samples per task for pass@k estimation.")
    export_jsonl: bool = Field(default=True)
    export_otlp: bool = Field(default=False)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PipelineConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(**data)

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
