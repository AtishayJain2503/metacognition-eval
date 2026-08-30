"""
nemo_eval.agents.vanilla
------------------------
Vanilla Zero-Shot Chain-of-Thought Execution Engine (Milestone 3).

Executes single-turn prompts without tool access, capturing raw completion,
extracting scalar values, recording real-time hardware telemetry, and
evaluating answer correctness.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.eval.math_eval import SympyMathEvaluator
from nemo_eval.models.base import BaseLLMClient, LLMMessage
from nemo_eval.telemetry.extractor import ValueExtractor
from nemo_eval.telemetry.tracer import EpisodeTrajectory, TrajectoryState, TrajectoryTracer


class BaseEvaluationEngine:
    """Abstract base evaluation engine protocol."""

    def evaluate_task(
        self,
        task: BenchmarkTask,
        model: Optional[BaseLLMClient] = None,
        **kwargs
    ) -> EpisodeTrajectory:
        raise NotImplementedError


class VanillaEngine(BaseEvaluationEngine):
    """
    Zero-Shot Chain-of-Thought Engine with 0 tool calls.

    Generates a single-turn response to the benchmark query, extracts the final
    answer using ValueExtractor, evaluates against ground truth, and tracks
    hardware metrics (duration, RAM, GPU VRAM, power, energy).
    """

    def __init__(
        self,
        model: Optional[BaseLLMClient] = None,
        enable_telemetry: bool = True,
        sample_interval_s: float = 0.02,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.enable_telemetry = enable_telemetry
        self.sample_interval_s = sample_interval_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate_task(
        self,
        task: BenchmarkTask,
        model: Optional[BaseLLMClient] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> EpisodeTrajectory:
        """
        Execute zero-shot CoT evaluation on a single benchmark task.

        Args:
            task: The BenchmarkTask instance to evaluate.
            model: Optional BaseLLMClient override.
            system_prompt: Optional custom system instructions.

        Returns:
            EpisodeTrajectory containing steps, metrics, extracted answer, and score.
        """
        active_model = model or self.model
        if active_model is None:
            raise ValueError("No LLM client provided to VanillaEngine.")

        model_name = getattr(active_model, "model_name", "unknown_model")
        tracer = TrajectoryTracer(
            task_id=task.task_id,
            model_name=model_name,
            enable_telemetry=self.enable_telemetry,
            sample_interval_s=self.sample_interval_s,
        )
        tracer.begin_episode()

        # Build prompt
        default_sys = (
            "You are a precise mathematical and analytical reasoning assistant. "
            "Solve the problem step-by-step using Chain-of-Thought reasoning. "
            "Write the final scalar or symbolic answer in LaTeX \\boxed{} format at the very end."
        )
        sys_msg = system_prompt or default_sys
        prob_text = task.problem_text if hasattr(task, "problem_text") and task.problem_text else getattr(task, "query", "")
        prompt = f"Problem: {prob_text}\nPlease solve step-by-step and write the final answer in \\boxed{{}}."

        # Step 1: ACTION_SELECTION
        tracer.transition(
            TrajectoryState.ACTION_SELECTION,
            input_payload={"mode": "vanilla", "prompt": prompt, "system": sys_msg},
        )

        raw_completion = ""
        try:
            # Model client invocation: supports both generate(messages=...) and generate(prompt=...)
            if hasattr(active_model, "generate"):
                try:
                    # Attempt structured LLMMessage invocation
                    messages = [
                        LLMMessage(role="system", content=sys_msg),
                        LLMMessage(role="user", content=prompt),
                    ]
                    response = active_model.generate(
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    if hasattr(response, "content"):
                        raw_completion = response.content or ""
                    elif isinstance(response, str):
                        raw_completion = response
                    else:
                        raw_completion = str(response)
                except TypeError:
                    # Direct string invocation
                    response = active_model.generate(prompt, system=sys_msg)
                    if hasattr(response, "content"):
                        raw_completion = response.content or ""
                    elif isinstance(response, str):
                        raw_completion = response
                    else:
                        raw_completion = str(response)
        except Exception as e:
            tracer.transition(
                TrajectoryState.TERMINAL_FAILURE,
                input_payload={"error": str(e)},
            )
            return tracer.close_episode(
                status="failed",
                final_answer=None,
                ground_truth_score=0.0,
                plan_adherence_score=1.0,
                tool_accuracy=1.0,
            )

        # Step 2: Answer extraction
        eval_type = getattr(task, "eval_type", "exact")
        extracted = ValueExtractor.extract_value(raw_completion, expected_type=eval_type)

        # Step 3: Ground truth evaluation
        score = 0.0
        try:
            eval_res = evaluate_task_result(task=task, candidate_output=extracted)
            score = float(eval_res.score)
        except Exception:
            try:
                score = float(SympyMathEvaluator.evaluate(
                    candidate=extracted,
                    ground_truth=task.ground_truth,
                    eval_type=eval_type,
                ))
            except Exception:
                score = 1.0 if str(extracted).strip() == str(task.ground_truth).strip() else 0.0

        # Step 4: FINAL_SYNTHESIS
        tracer.transition(
            TrajectoryState.FINAL_SYNTHESIS,
            output_payload={
                "raw_completion": raw_completion,
                "extracted_answer": extracted,
                "score": score,
            },
        )

        status = "success" if score == 1.0 else "failed"
        tracer.transition(
            TrajectoryState.TERMINAL_SUCCESS if status == "success" else TrajectoryState.TERMINAL_FAILURE
        )

        traj = tracer.close_episode(
            status=status,
            final_answer=extracted,
            ground_truth_score=score,
            plan_adherence_score=1.0,
            tool_accuracy=1.0,
            spea=1.0,
        )
        return traj
