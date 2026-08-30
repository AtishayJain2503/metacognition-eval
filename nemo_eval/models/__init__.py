"""
nemo_eval.models
----------------
Model provider interface abstractions, client implementations, and deterministic mock runners.
"""

from nemo_eval.models.base import (
    BaseLLMClient,
    ChatMessage,
    FunctionCall,
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMInvalidResponseError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    ModelConfig,
    ModelRegistry,
    ToolCall,
    get_model_client,
)
from nemo_eval.models.groq import GroqLLMClient, extract_think_reasoning
from nemo_eval.models.mock_runner import DeterministicMockLLMClient
from nemo_eval.models.nemo_client import NeMoClient, extract_nemo_special_tokens
from nemo_eval.models.openai_gateway import OpenAIGatewayClient, extract_text_fallback_tool_calls
from nemo_eval.models.registry import (
    MODEL_CONFIGS,
    MODEL_FAMILIES,
    TARGET_MODELS,
    TargetModelSpec,
    create_target_model_client,
    get_model_spec,
    get_target_models,
)

# Register standard providers in ModelRegistry
ModelRegistry.register("groq", GroqLLMClient)
ModelRegistry.register("openai", OpenAIGatewayClient)
ModelRegistry.register("openai_gateway", OpenAIGatewayClient)
ModelRegistry.register("vllm", OpenAIGatewayClient)
ModelRegistry.register("ollama", OpenAIGatewayClient)
ModelRegistry.register("tgi", OpenAIGatewayClient)
ModelRegistry.register("sglang", OpenAIGatewayClient)
ModelRegistry.register("together", OpenAIGatewayClient)
ModelRegistry.register("nemo", NeMoClient)
ModelRegistry.register("nim", NeMoClient)
ModelRegistry.register("mock", DeterministicMockLLMClient)
ModelRegistry.register("mock_runner", DeterministicMockLLMClient)

__all__ = [
    "BaseLLMClient",
    "ModelConfig",
    "ChatMessage",
    "LLMMessage",
    "LLMResponse",
    "ToolCall",
    "FunctionCall",
    "ModelRegistry",
    "get_model_client",
    "GroqLLMClient",
    "extract_think_reasoning",
    "OpenAIGatewayClient",
    "extract_text_fallback_tool_calls",
    "NeMoClient",
    "extract_nemo_special_tokens",
    "DeterministicMockLLMClient",
    "LLMProviderError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMContextLengthExceededError",
    "LLMInvalidResponseError",
    # Target models registry
    "TARGET_MODELS",
    "MODEL_FAMILIES",
    "MODEL_CONFIGS",
    "TargetModelSpec",
    "get_target_models",
    "get_model_spec",
    "create_target_model_client",
]
