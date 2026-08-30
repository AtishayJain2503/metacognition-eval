"""
nemo_eval.eval.base
===================
Standardized evaluation result data structures and evaluator protocols.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvalResult(BaseModel):
    """Standardized evaluation verdict envelope across all evaluation engines."""
    model_config = ConfigDict(extra="ignore")

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized correctness score between 0.0 and 1.0."
    )
    is_correct: bool = Field(
        ..., description="Binary verdict indicating whether candidate passed ground truth."
    )
    eval_type: str = Field(
        ..., description="Evaluation strategy applied (exact, float_tol, sql_multiset, dataframe_diff)."
    )
    candidate_parsed: Any = Field(
        default=None, description="Extracted and parsed candidate output."
    )
    gold_target: Any = Field(
        default=None, description="Target ground truth reference."
    )
    diagnostic_message: str = Field(
        default="", description="Detailed diagnostic report or failure reason."
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Evaluation execution duration in milliseconds."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Supplementary comparison telemetry."
    )
