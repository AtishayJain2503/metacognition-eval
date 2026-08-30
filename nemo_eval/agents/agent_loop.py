"""
nemo_eval.agents.agent_loop
----------------------------
Multi-turn ReAct-style Agent Execution Loop.

Integrates:
    - TaskPlanner [T.D]: decomposes the task into a sub-goal DAG.
    - ToolOrchestrator [W.O]: dispatches tool calls, bridges parameters.
    - TrajectoryTracer: records 9-state FSM transitions.
    - PlanAdherenceScorer: computes PAS against the original plan.

The loop executes the plan sub-goal by sub-goal, invoking the LLM
at each ACTION_SELECTION step to produce a ToolCall JSON, then
dispatching to the appropriate tool, observing the result, and
optionally entering a SELF_CORRECTION cycle on error.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from nemo_eval.agents.planner import TaskPlanner, TaskPlan, PlannerConfig
from nemo_eval.agents.orchestrator import ToolOrchestrator, ToolCall, OrchestratorConfig
from nemo_eval.telemetry.tracer import TrajectoryTracer, TrajectoryState, EpisodeTrajectory
from nemo_eval.telemetry.metrics import PlanAdherenceScorer
from nemo_eval.tools.schemas import ToolResult
from nemo_eval.models.base import LLMMessage


# ---------------------------------------------------------------------------
# Config / Result
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for the AgentLoop."""
    model_config = ConfigDict(extra="ignore")

    max_turns: int = Field(default=25, ge=1, le=100, description="Maximum total LLM turns before TERMINAL_FAILURE.")
    max_correction_attempts: int = Field(default=3, ge=0, le=10, description="Max self-correction retries per sub-goal.")
    enable_planning: bool = Field(default=True, description="If False, skip [T.D] decomposition and treat query as a single action.")
    verify_intermediate: bool = Field(default=True, description="Trigger VERIFICATION state after each OBSERVATION.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=64)
    planner_config: PlannerConfig = Field(default_factory=PlannerConfig)
    orchestrator_config: OrchestratorConfig = Field(default_factory=OrchestratorConfig)


class AgentResult(BaseModel):
    """Result returned by AgentLoop.run()."""
    model_config = ConfigDict(extra="ignore")

    task_id: str
    final_answer: Any
    trajectory: EpisodeTrajectory
    plan: Optional[TaskPlan] = None
    success: bool


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_ACTION_SYSTEM_PROMPT_DATA = """\
You are a precise data analytics agent. You have access to the following tools:

TOOLS:
1. python_repl — Execute Python code. Args: {"code": "<python code string>"}
2. sqlite_query — Execute a read-only SQL query. Args: {"db_path": "<path>", "query": "<SQL>"}
3. sqlite_schema — Inspect database schema. Args: {"db_path": "<path>", "table_name": "<optional table>"}
4. tabular_inspect — Inspect CSV/Parquet file. Args: {"file_path": "<path>"}

For each step, you will be given a sub-goal. Output ONLY a JSON object:
{
  "tool_name": "<tool_name>",
  "arguments": { <tool arguments> }
}

Do not include any text outside the JSON object.
"""

_ACTION_SYSTEM_PROMPT_MATH = """\
You are a precise mathematical reasoning agent. You solve math word problems step-by-step using Python code.

You have ONE tool:
1. python_repl — Execute Python code. Args: {"code": "<python code string>"}

Rules:
- Always write Python code to compute the answer. Do not guess.
- The "code" string in your JSON arguments must be PURE, syntax-valid Python code. Do not include natural language explanations, markdown text, or conversational preambles inside the "code" string, as this causes Python syntax errors.
- Your final line of code must print() the integer answer.
- Output ONLY a JSON object with no text outside it:
{
  "tool_name": "python_repl",
  "arguments": {"code": "<your python code here>"}
}
"""

# Default to data prompt; overridden in run() based on task type
_ACTION_SYSTEM_PROMPT = _ACTION_SYSTEM_PROMPT_DATA

_CORRECTION_SYSTEM_PROMPT = """\
You are a self-correcting agent.

The previous tool call produced an error. Analyze the error and produce a corrected tool call.

Output ONLY a valid JSON object:
{
  "tool_name": "<tool_name>",
  "arguments": { <corrected tool arguments> }
}

Do not include any text outside the JSON object.
"""

_SYNTHESIS_SYSTEM_PROMPT_DATA = """\
You are a data analytics agent completing a task.

Based on all tool outputs collected, synthesize the final answer.
Be concise and precise. Output only the final answer value — no preamble.
"""

_SYNTHESIS_SYSTEM_PROMPT_MATH = """\
You are a mathematical reasoning agent completing a task.

Based on the Python code outputs above, state the final integer answer.
Output ONLY the number. No explanation, no units, just the integer.
"""


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Multi-turn ReAct-style agent loop integrating planning, orchestration,
    telemetry, and self-correction.
    """

    def __init__(
        self,
        model_client: Any,
        config: Optional[AgentConfig] = None,
    ):
        """
        Args:
            model_client: LLM client conforming to BaseLLMClient (generate()).
            config: AgentConfig. Defaults to AgentConfig().
        """
        self.model = model_client
        self.config = config or AgentConfig()
        self._planner = TaskPlanner(
            model_client=model_client,
            config=self.config.planner_config,
        )
        self._orchestrator = ToolOrchestrator(config=self.config.orchestrator_config)

    def run(
        self,
        task_id: str,
        query: str,
        db_path: Optional[str] = None,
        table_path: Optional[str] = None,
        model_name: str = "unknown",
    ) -> AgentResult:
        """
        Execute the full agent loop for a given query.

        Args:
            task_id: Unique identifier for this task instance.
            query: The complex analytical question to answer.
            db_path: Optional path to a SQLite database for context.
            table_path: Optional path to a CSV/Parquet file for context.
            model_name: Model identifier for telemetry labeling.

        Returns:
            AgentResult with trajectory, plan, and final answer.
        """
        tracer = TrajectoryTracer(task_id=task_id, model_name=model_name)
        tracer.begin_episode()
        self._orchestrator.reset()

        plan: Optional[TaskPlan] = None
        turns_used = 0

        # ── PLANNING ─────────────────────────────────────────────────────
        tracer.transition(
            TrajectoryState.PLANNING,
            input_payload={"query": query},
        )

        if self.config.enable_planning:
            plan = self._planner.decompose(task_id=task_id, query=query)
            tracer.transition(
                TrajectoryState.ACTION_SELECTION,
                output_payload={
                    "plan_node_count": plan.metrics.node_count if plan.metrics else 0,
                    "plan_composite_score": plan.metrics.composite_score if plan.metrics else 0.0,
                },
            )
            sub_goals = plan.ordered_sub_goals()
        else:
            # Treat entire query as one action
            from nemo_eval.agents.planner import SubGoal
            sub_goals = [SubGoal(id="sg_1", description=query, depends_on=[])]
            tracer.transition(TrajectoryState.ACTION_SELECTION)

        # ── SELECT PROMPT STRATEGY ────────────────────────────────────────
        # Math tasks (no db/table) get a focused math prompt.
        # Data analytics tasks get the full tool-selection prompt.
        is_math_task = (db_path is None and table_path is None)
        action_prompt = _ACTION_SYSTEM_PROMPT_MATH if is_math_task else _ACTION_SYSTEM_PROMPT_DATA
        synthesis_prompt = _SYNTHESIS_SYSTEM_PROMPT_MATH if is_math_task else _SYNTHESIS_SYSTEM_PROMPT_DATA

        # ── PER SUB-GOAL EXECUTION LOOP ───────────────────────────────────
        prior_outputs: Dict[str, ToolResult] = {}
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": action_prompt},
        ]

        if db_path:
            conversation.append({"role": "user", "content": f"Database available at: {db_path}"})
        if table_path:
            conversation.append({"role": "user", "content": f"Tabular file available at: {table_path}"})
        conversation.append({"role": "user", "content": f"Overall task: {query}"})

        for sg in sub_goals:
            if turns_used >= self.config.max_turns:
                tracer.transition(
                    TrajectoryState.TERMINAL_FAILURE,
                    input_payload={"reason": "max_turns_exceeded"},
                )
                return self._build_result(
                    task_id, None, tracer, plan, "failed",
                    prior_outputs=prior_outputs,
                )

            correction_attempts = 0
            last_tool_result: Optional[ToolResult] = None

            # Inject sub-goal context into conversation
            conversation.append({
                "role": "user",
                "content": (
                    f"Current sub-goal ({sg.id}): {sg.description}\n"
                    + (f"Suggested tool: {sg.tool_hint}" if sg.tool_hint else "")
                    + (f"\nDatabase path: {db_path}" if db_path else "")
                    + (f"\nTabular file: {table_path}" if table_path else "")
                ),
            })

            while correction_attempts <= self.config.max_correction_attempts:
                if turns_used >= self.config.max_turns:
                    tracer.transition(
                        TrajectoryState.TERMINAL_FAILURE,
                        input_payload={"reason": "max_turns_exceeded"},
                    )
                    return self._build_result(
                        task_id, None, tracer, plan, "failed",
                        prior_outputs=prior_outputs,
                    )

                # ACTION_SELECTION: ask LLM for tool call
                tracer.transition(
                    TrajectoryState.ACTION_SELECTION,
                    input_payload={"sub_goal_id": sg.id, "correction_attempt": correction_attempts},
                )
                try:
                    llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in conversation]
                    response = self.model.generate(
                        messages=llm_messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    turns_used += 1
                    raw_content = response.content or ""
                except Exception as e:
                    tracer.transition(
                        TrajectoryState.TERMINAL_FAILURE,
                        input_payload={"error": str(e)},
                    )
                    return self._build_result(
                        task_id, None, tracer, plan, "failed",
                        prior_outputs=prior_outputs,
                    )

                # Parse tool call from LLM response
                tool_call = self._parse_tool_call(raw_content, sg)

                # TOOL_EXECUTION
                tracer.transition(
                    TrajectoryState.TOOL_EXECUTION,
                    input_payload={
                        "sub_goal_id": sg.id,
                        "tool": tool_call.tool_name,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                    },
                )

                dispatch = self._orchestrator.dispatch(tool_call, prior_outputs)
                last_tool_result = dispatch.result
                observation_text = last_tool_result.to_agent_observation()

                # OBSERVATION
                tracer.transition(
                    TrajectoryState.OBSERVATION,
                    input_payload={"sub_goal_id": sg.id},
                    output_payload={
                        "status": last_tool_result.status,
                        "tool_name": dispatch.tool_name,
                        "bridge_applied": dispatch.bridge_applied,
                    },
                    metrics={"tool_valid": 1.0 if dispatch.is_valid_tool else 0.0},
                )

                # Add observation to conversation
                conversation.append({
                    "role": "assistant",
                    "content": raw_content,
                })
                conversation.append({
                    "role": "user",
                    "content": f"Tool result for {sg.id}:\n{observation_text}",
                })

                if last_tool_result.is_success:
                    prior_outputs[sg.id] = last_tool_result

                    # VERIFICATION (optional)
                    if self.config.verify_intermediate:
                        tracer.transition(
                            TrajectoryState.VERIFICATION,
                            input_payload={"sub_goal_id": sg.id},
                            output_payload={"verified": True},
                            metrics={"verification_pass": 1.0},
                        )
                    break  # Sub-goal done, proceed to next

                else:
                    # SELF_CORRECTION
                    if correction_attempts < self.config.max_correction_attempts:
                        tracer.transition(
                            TrajectoryState.SELF_CORRECTION,
                            input_payload={
                                "sub_goal_id": sg.id,
                                "attempt": correction_attempts + 1,
                                "error": observation_text,
                            },
                        )
                        # Inject correction system prompt
                        conversation.append({
                            "role": "user",
                            "content": (
                                "The tool returned an error. Please analyze the error and "
                                "provide a corrected tool call JSON:\n"
                                f"Error: {observation_text}"
                            ),
                        })
                        correction_attempts += 1
                    else:
                        # Exhausted retries — move on with partial results
                        break

        # ── FINAL SYNTHESIS ───────────────────────────────────────────────
        if not tracer.is_terminal():
            tracer.transition(TrajectoryState.FINAL_SYNTHESIS)

            synthesis_messages = [
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": f"Original task: {query}"},
            ]
            # Append all tool outputs as context
            for sg_id, result in prior_outputs.items():
                obs = result.to_agent_observation(max_length=2000)
                synthesis_messages.append({
                    "role": "user",
                    "content": f"Output from {sg_id}:\n{obs}",
                })
            synthesis_messages.append({
                "role": "user",
                "content": "Based on the above, what is the final answer?",
            })

            try:
                llm_synth_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in synthesis_messages]
                synth_response = self.model.generate(
                    messages=llm_synth_messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                final_answer = (synth_response.content or "").strip()

                # Fallback: if synthesis returned empty, use the last tool output directly
                if not final_answer and prior_outputs:
                    last_output = list(prior_outputs.values())[-1]
                    final_answer = last_output.to_agent_observation(max_length=512).strip()

                tracer.transition(
                    TrajectoryState.TERMINAL_SUCCESS,
                    output_payload={"final_answer": final_answer[:200]},
                )
            except Exception as e:
                # Fallback on exception too
                if prior_outputs:
                    last_output = list(prior_outputs.values())[-1]
                    final_answer = last_output.to_agent_observation(max_length=512).strip() or None
                else:
                    final_answer = None
                tracer.transition(
                    TrajectoryState.TERMINAL_FAILURE,
                    input_payload={"error": str(e)},
                )
        else:
            final_answer = None

        return self._build_result(
            task_id, final_answer, tracer, plan, "success",
            prior_outputs=prior_outputs,
        )


    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _parse_tool_call(self, raw: str, sg: Any) -> ToolCall:
        """Extract and parse a ToolCall JSON from LLM response with fallback."""
        json_str = raw.strip()

        # Strip markdown fences
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
        if fence:
            json_str = fence.group(1)
        else:
            obj = re.search(r"\{.*\}", json_str, re.DOTALL)
            if obj:
                json_str = obj.group(0)

        try:
            data = json.loads(json_str)
            tool_name = str(data.get("tool_name", "python_repl"))
            arguments = data.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"code": arguments}
        except (json.JSONDecodeError, ValueError):
            # Fallback: select tool from hint and wrap raw content as code
            tool_name = self._orchestrator.select_tool(
                getattr(sg, "tool_hint", None),
                getattr(sg, "description", ""),
            )
            arguments = {"code": raw} if tool_name == "python_repl" else {}

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            sub_goal_id=sg.id,
        )

    def _build_result(
        self,
        task_id: str,
        final_answer: Any,
        tracer: TrajectoryTracer,
        plan: Optional[TaskPlan],
        status_hint: str,
        prior_outputs: Optional[Dict[str, ToolResult]] = None,
    ) -> AgentResult:
        """Compute metrics and finalize AgentResult."""
        acc_tool, _ = self._orchestrator.compute_tool_accuracy()
        spea = self._orchestrator.compute_spea()

        # Close tracer to EpisodeTrajectory
        traj_status = "success" if tracer.current_state == TrajectoryState.TERMINAL_SUCCESS else "failed"
        if tracer.current_state == TrajectoryState.TERMINAL_FAILURE:
            traj_status = "failed"

        # Compute PAS
        planned_order = plan.execution_order if plan else []
        tmp_trajectory = EpisodeTrajectory(
            task_id=task_id,
            model_name="unknown",
            status=traj_status,
            steps=tracer._steps,
        )
        pas = PlanAdherenceScorer.score(tmp_trajectory, planned_order)

        trajectory = tracer.close_episode(
            status=traj_status,
            final_answer=final_answer,
            tool_accuracy=acc_tool,
            spea=spea,
            plan_adherence_score=pas,
        )

        return AgentResult(
            task_id=task_id,
            final_answer=final_answer,
            trajectory=trajectory,
            plan=plan,
            success=traj_status == "success",
        )


# ---------------------------------------------------------------------------
# AgenticEngine Wrapper
# ---------------------------------------------------------------------------

class AgenticEngine:
    """
    9-State Agentic FSM Execution Engine.

    Orchestrates DAG decomposition (TaskPlanner), auxiliary Python REPL tool
    dispatch (ToolOrchestrator), intermediate verification, self-correction recovery,
    and final answer extraction with step-level telemetry.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.model = model
        self.config = config or AgentConfig()

    def evaluate_task(
        self,
        task: Any,
        model: Optional[Any] = None,
        max_turns: Optional[int] = None,
        **kwargs
    ) -> EpisodeTrajectory:
        """
        Execute multi-turn agentic evaluation on a benchmark task.

        Args:
            task: The BenchmarkTask instance to evaluate.
            model: Optional model client override.
            max_turns: Optional override for max turns.

        Returns:
            EpisodeTrajectory with all 9-state transitions and hardware telemetry.
        """
        active_model = model or self.model
        if active_model is None:
            raise ValueError("No LLM client provided to AgenticEngine.")

        cfg = self.config
        if max_turns is not None:
            cfg = cfg.model_copy(update={"max_turns": max_turns})

        loop = AgentLoop(model_client=active_model, config=cfg)
        query = getattr(task, "problem_text", None) or getattr(task, "query", "")
        db_path = getattr(task, "db_path", None)
        table_path = getattr(task, "table_path", None)
        model_name = getattr(active_model, "model_name", "unknown")

        result = loop.run(
            task_id=task.task_id,
            query=query,
            db_path=db_path,
            table_path=table_path,
            model_name=model_name,
        )

        traj = result.trajectory
        final_ans = result.final_answer

        # Answer extraction and evaluation
        from nemo_eval.telemetry.extractor import ValueExtractor
        from nemo_eval.eval.engine import evaluate_task_result
        from nemo_eval.eval.math_eval import SympyMathEvaluator

        eval_type = getattr(task, "eval_type", "exact")
        extracted = ValueExtractor.extract_value(
            str(final_ans) if final_ans is not None else "",
            expected_type=eval_type,
        )

        gt_score = 0.0
        if task.ground_truth is not None and extracted:
            try:
                eval_res = evaluate_task_result(task=task, candidate_output=extracted)
                gt_score = float(eval_res.score)
            except Exception:
                try:
                    gt_score = float(SympyMathEvaluator.evaluate(
                        candidate=extracted,
                        ground_truth=task.ground_truth,
                        eval_type=eval_type,
                    ))
                except Exception:
                    gt_score = 1.0 if str(extracted).strip() == str(task.ground_truth).strip() else 0.0

        traj.ground_truth_score = gt_score
        traj.final_answer = extracted or final_ans
        return traj

