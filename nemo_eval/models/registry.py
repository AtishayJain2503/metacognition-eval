"""
nemo_eval.models.registry
-------------------------
Target Models Registry & Benchmark Configurations (Milestone 4).

Configures all 7 target models:
1. Qwen2.5-Math-7B
2. DeepSeek-R1-7B
3. Phi4-mini-reasoning
4. Llama3.2-3B
5. Qwen2.5-Math-1.5B
6. DeepSeek-R1-1.5B
7. Qwen3-4B-Thinking

Supports:
- Family classifications (Qwen, DeepSeek, Phi, Llama)
- Generation hyperparameter presets (temperature, max_tokens, top_p)
- DeepSeek-R1 <think> token isolation
- Client instantiation factory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from nemo_eval.models.base import BaseLLMClient, ModelConfig, ModelRegistry, get_model_client


TARGET_MODELS: List[str] = [
    "Qwen2.5-Math-7B",
    "DeepSeek-R1-7B",
    "Phi4-mini-reasoning",
    "Llama3.2-3B",
    "Qwen2.5-Math-1.5B",
    "DeepSeek-R1-1.5B",
    "Qwen3-4B-Thinking",
]

MODEL_FAMILIES: Dict[str, str] = {
    "Qwen2.5-Math-7B": "Qwen",
    "DeepSeek-R1-7B": "DeepSeek",
    "Phi4-mini-reasoning": "Phi",
    "Llama3.2-3B": "Llama",
    "Qwen2.5-Math-1.5B": "Qwen",
    "DeepSeek-R1-1.5B": "DeepSeek",
    "Qwen3-4B-Thinking": "Qwen",
}

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Qwen2.5-Math-7B": {
        "family": "Qwen",
        "parameters": "7B",
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "is_reasoning_model": False,
        "think_isolation": False,
        "default_provider": "vllm",
    },
    "DeepSeek-R1-7B": {
        "family": "DeepSeek",
        "parameters": "7B",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.95,
        "is_reasoning_model": True,
        "think_isolation": True,
        "default_provider": "groq",
    },
    "Phi4-mini-reasoning": {
        "family": "Phi",
        "parameters": "3.8B",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.95,
        "is_reasoning_model": True,
        "think_isolation": False,
        "default_provider": "openai_gateway",
    },
    "Llama3.2-3B": {
        "family": "Llama",
        "parameters": "3B",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.9,
        "is_reasoning_model": False,
        "think_isolation": False,
        "default_provider": "groq",
    },
    "Qwen2.5-Math-1.5B": {
        "family": "Qwen",
        "parameters": "1.5B",
        "temperature": 0.0,
        "max_tokens": 4096,
        "top_p": 1.0,
        "is_reasoning_model": False,
        "think_isolation": False,
        "default_provider": "vllm",
    },
    "DeepSeek-R1-1.5B": {
        "family": "DeepSeek",
        "parameters": "1.5B",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.95,
        "is_reasoning_model": True,
        "think_isolation": True,
        "default_provider": "groq",
    },
    "Qwen3-4B-Thinking": {
        "family": "Qwen",
        "parameters": "4B",
        "temperature": 0.6,
        "max_tokens": 4096,
        "top_p": 0.95,
        "is_reasoning_model": True,
        "think_isolation": True,
        "default_provider": "openai_gateway",
    },
}


class TargetModelSpec(BaseModel):
    """Specification model for a target benchmark model."""
    model_config = ConfigDict(extra="ignore")

    model_name: str
    family: str
    parameters: str
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    is_reasoning_model: bool = False
    think_isolation: bool = False
    default_provider: str = "mock"


def get_target_models() -> List[str]:
    """Return the list of all 7 target benchmark models."""
    return list(TARGET_MODELS)


def get_model_spec(model_name: str) -> TargetModelSpec:
    """Retrieve full TargetModelSpec for a model name."""
    if model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        return TargetModelSpec(model_name=model_name, **cfg)
    
    # Fallback heuristic
    fam = "Custom"
    for m, f in MODEL_FAMILIES.items():
        if f.lower() in model_name.lower():
            fam = f
            break
    is_r1 = "r1" in model_name.lower() or "thinking" in model_name.lower()
    return TargetModelSpec(
        model_name=model_name,
        family=fam,
        parameters="Unknown",
        temperature=0.6 if is_r1 else 0.0,
        max_tokens=4096,
        is_reasoning_model=is_r1,
        think_isolation=is_r1,
        default_provider="mock",
    )


def create_target_model_client(
    model_name: str,
    provider: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Instantiate an LLM client configured for one of the target models.
    """
    spec = get_model_spec(model_name)
    active_provider = provider or spec.default_provider or "mock"
    client_kwargs = {
        "model_name": model_name,
        "temperature": kwargs.get("temperature", spec.temperature),
        "max_tokens": kwargs.get("max_tokens", spec.max_tokens),
        "top_p": kwargs.get("top_p", spec.top_p),
        **kwargs,
    }
    return get_model_client(active_provider, **client_kwargs)
