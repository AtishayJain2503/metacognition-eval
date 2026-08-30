"""
nemo_eval.telemetry
-------------------
Multi-Turn 9-State Trajectory FSM and Telemetry Logger (Milestone 4).

Exports:
    - TrajectoryState: Enum of 9 FSM states.
    - StepEvent: Per-turn event record.
    - EpisodeTrajectory: Complete episode record.
    - TrajectoryTracer: FSM state machine and event logger.
    - PlanAdherenceScorer: PAS metric calculator.
    - TelemetryExporter: JSONL, Markdown, and OTLP exporters.
"""

from nemo_eval.telemetry.tracer import (
    TrajectoryState,
    StepEvent,
    EpisodeTrajectory,
    TrajectoryTracer,
)
from nemo_eval.telemetry.metrics import PlanAdherenceScorer
from nemo_eval.telemetry.exporters import TelemetryExporter

__all__ = [
    "TrajectoryState",
    "StepEvent",
    "EpisodeTrajectory",
    "TrajectoryTracer",
    "PlanAdherenceScorer",
    "TelemetryExporter",
]
