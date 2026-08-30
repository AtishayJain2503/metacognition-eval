"""
nemo_eval.telemetry.tracer
--------------------------
9-State Trajectory Finite State Machine (FSM) and per-step event logger.

FSM State Diagram:
    PLANNING
        → ACTION_SELECTION
            → TOOL_EXECUTION
                → OBSERVATION
                    → VERIFICATION
                        → SELF_CORRECTION → ACTION_SELECTION (retry loop)
                        → FINAL_SYNTHESIS
                            → TERMINAL_SUCCESS
                            → TERMINAL_FAILURE
                    → FINAL_SYNTHESIS (skip verification)
    Any state → TERMINAL_FAILURE (on max_turns or hard error)

Legal Transitions:
    PLANNING             → {ACTION_SELECTION, TERMINAL_FAILURE}
    ACTION_SELECTION     → {TOOL_EXECUTION, FINAL_SYNTHESIS, TERMINAL_FAILURE}
    TOOL_EXECUTION       → {OBSERVATION, TERMINAL_FAILURE}
    OBSERVATION          → {VERIFICATION, FINAL_SYNTHESIS, TERMINAL_FAILURE}
    VERIFICATION         → {SELF_CORRECTION, FINAL_SYNTHESIS, TERMINAL_FAILURE}
    SELF_CORRECTION      → {ACTION_SELECTION, TERMINAL_FAILURE}
    FINAL_SYNTHESIS      → {TERMINAL_SUCCESS, TERMINAL_FAILURE}
    TERMINAL_SUCCESS     → {}
    TERMINAL_FAILURE     → {}
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# State Enum
# ---------------------------------------------------------------------------

class TrajectoryState(str, Enum):
    PLANNING = "PLANNING"
    ACTION_SELECTION = "ACTION_SELECTION"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    OBSERVATION = "OBSERVATION"
    VERIFICATION = "VERIFICATION"
    SELF_CORRECTION = "SELF_CORRECTION"
    FINAL_SYNTHESIS = "FINAL_SYNTHESIS"
    TERMINAL_SUCCESS = "TERMINAL_SUCCESS"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


# FSM legal transitions
_TRANSITIONS: Dict[TrajectoryState, Set[TrajectoryState]] = {
    TrajectoryState.PLANNING: {
        TrajectoryState.ACTION_SELECTION,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.ACTION_SELECTION: {
        TrajectoryState.TOOL_EXECUTION,
        TrajectoryState.FINAL_SYNTHESIS,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.TOOL_EXECUTION: {
        TrajectoryState.OBSERVATION,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.OBSERVATION: {
        TrajectoryState.VERIFICATION,
        TrajectoryState.FINAL_SYNTHESIS,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.VERIFICATION: {
        TrajectoryState.SELF_CORRECTION,
        TrajectoryState.FINAL_SYNTHESIS,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.SELF_CORRECTION: {
        TrajectoryState.ACTION_SELECTION,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.FINAL_SYNTHESIS: {
        TrajectoryState.TERMINAL_SUCCESS,
        TrajectoryState.TERMINAL_FAILURE,
    },
    TrajectoryState.TERMINAL_SUCCESS: set(),
    TrajectoryState.TERMINAL_FAILURE: set(),
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class StepEvent(BaseModel):
    """A single step event in the multi-turn trajectory log."""
    model_config = ConfigDict(extra="ignore")

    step_id: int = Field(..., ge=0)
    state: TrajectoryState
    timestamp: float = Field(..., description="Unix epoch seconds at step entry.")
    duration_ms: float = Field(default=0.0, ge=0.0)
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    output_payload: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-step float metrics: e.g. plan_adherence, tool_validity, error_recovered.",
    )
    invalid_transition: bool = Field(
        default=False,
        description="True if this step violated FSM legal transition rules.",
    )


class EpisodeTrajectory(BaseModel):
    """Complete multi-turn execution record for a single task episode."""
    model_config = ConfigDict(extra="ignore")

    task_id: str
    model_name: str
    status: Literal["success", "failed", "timeout", "max_turns_exceeded"] = "failed"
    steps: List[StepEvent] = Field(default_factory=list)
    total_duration_ms: float = 0.0

    # Summary metrics (populated at episode close)
    plan_adherence_score: float = 0.0
    tool_accuracy: float = 0.0
    spea: float = 0.0
    self_correction_attempts: int = 0
    self_correction_success: bool = False
    invalid_transitions: int = 0

    final_answer: Any = None
    ground_truth_score: float = 0.0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("success", "failed", "timeout", "max_turns_exceeded")

    def state_sequence(self) -> List[str]:
        return [s.state.value for s in self.steps]


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class TrajectoryTracer:
    """
    9-State FSM trajectory machine.

    Tracks agent lifecycle, records per-step events with metrics,
    detects illegal transitions, and finalizes EpisodeTrajectory.

    Usage:
        tracer = TrajectoryTracer(task_id="t1", model_name="llama-3.3-70b")
        tracer.begin_episode()
        tracer.transition(TrajectoryState.ACTION_SELECTION, input_payload={...})
        ...
        trajectory = tracer.close_episode(status="success", final_answer=result)
    """

    def __init__(self, task_id: str, model_name: str = "unknown"):
        self.task_id = task_id
        self.model_name = model_name
        self._current_state: Optional[TrajectoryState] = None
        self._steps: List[StepEvent] = []
        self._step_id: int = 0
        self._episode_start: float = 0.0
        self._last_step_time: float = 0.0
        self._invalid_transitions: int = 0
        self._self_correction_attempts: int = 0
        self._self_correction_successes: int = 0

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def begin_episode(self) -> None:
        """Initialize a new episode, resetting all state."""
        self._current_state = None
        self._steps = []
        self._step_id = 0
        self._episode_start = time.monotonic()
        self._last_step_time = self._episode_start
        self._invalid_transitions = 0
        self._self_correction_attempts = 0
        self._self_correction_successes = 0

    def transition(
        self,
        new_state: TrajectoryState,
        input_payload: Optional[Dict[str, Any]] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> StepEvent:
        """
        Transition FSM to new_state, record a StepEvent.

        Illegal transitions are allowed but flagged in StepEvent.invalid_transition
        and counted in self._invalid_transitions.

        Returns the recorded StepEvent.
        """
        now = time.monotonic()
        duration_ms = (now - self._last_step_time) * 1000.0

        # Check transition legality
        is_invalid = False
        if self._current_state is not None:
            allowed = _TRANSITIONS.get(self._current_state, set())
            if new_state not in allowed:
                is_invalid = True
                self._invalid_transitions += 1

        # Track self-correction
        if new_state == TrajectoryState.SELF_CORRECTION:
            self._self_correction_attempts += 1

        event = StepEvent(
            step_id=self._step_id,
            state=new_state,
            timestamp=time.time(),
            duration_ms=round(duration_ms, 3),
            input_payload=input_payload or {},
            output_payload=output_payload or {},
            metrics=metrics or {},
            invalid_transition=is_invalid,
        )
        self._steps.append(event)
        self._step_id += 1
        self._current_state = new_state
        self._last_step_time = now

        return event

    def record_correction_success(self) -> None:
        """Mark that the last self-correction attempt led to recovery."""
        self._self_correction_successes += 1

    def close_episode(
        self,
        status: Literal["success", "failed", "timeout", "max_turns_exceeded"],
        final_answer: Any = None,
        ground_truth_score: float = 0.0,
        tool_accuracy: float = 1.0,
        spea: float = 1.0,
        plan_adherence_score: float = 0.0,
    ) -> EpisodeTrajectory:
        """
        Finalize and return the EpisodeTrajectory.

        Args:
            status: Terminal status of the episode.
            final_answer: The agent's final answer.
            ground_truth_score: Score from eval engine [0, 1].
            tool_accuracy: Acc_tool from ToolOrchestrator.
            spea: SPEA from ToolOrchestrator.
            plan_adherence_score: PAS from PlanAdherenceScorer.
        """
        total_ms = (time.monotonic() - self._episode_start) * 1000.0

        return EpisodeTrajectory(
            task_id=self.task_id,
            model_name=self.model_name,
            status=status,
            steps=list(self._steps),
            total_duration_ms=round(total_ms, 2),
            plan_adherence_score=round(plan_adherence_score, 4),
            tool_accuracy=round(tool_accuracy, 4),
            spea=round(spea, 4),
            self_correction_attempts=self._self_correction_attempts,
            self_correction_success=self._self_correction_successes > 0,
            invalid_transitions=self._invalid_transitions,
            final_answer=final_answer,
            ground_truth_score=round(ground_truth_score, 4),
        )

    @property
    def current_state(self) -> Optional[TrajectoryState]:
        return self._current_state

    @property
    def step_count(self) -> int:
        return self._step_id

    def is_terminal(self) -> bool:
        return self._current_state in (
            TrajectoryState.TERMINAL_SUCCESS,
            TrajectoryState.TERMINAL_FAILURE,
        )
