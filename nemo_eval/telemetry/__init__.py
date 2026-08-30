"""
nemo_eval.telemetry
-------------------
Hardware Resource Telemetry, Multi-Turn 9-State Trajectory FSM, and Value-Only Answer Extractor.

Exports:
    - HardwareMetrics: Dataclass capturing RAM, GPU VRAM, power, energy, and GPU availability.
    - HardwareMonitor: Background thread monitor for hardware resource telemetry.
    - ValueExtractor: Multi-tier target scalar/expression extractor.
    - TrajectoryState: Enum of 9 FSM states.
    - StepEvent: Per-step event record with hardware telemetry.
    - EpisodeTrajectory: Complete episode trajectory record with aggregated telemetry.
    - TrajectoryTracer: FSM state machine and event logger with telemetry sampling.
    - PlanAdherenceScorer: PAS metric calculator.
    - TelemetryExporter: JSONL, Markdown, and OTLP exporters.
"""

from nemo_eval.telemetry.monitor import (
    HardwareMetrics,
    HardwareMonitor,
)
from nemo_eval.telemetry.extractor import ValueExtractor
from nemo_eval.telemetry.tracer import (
    TrajectoryState,
    StepEvent,
    EpisodeTrajectory,
    TrajectoryTracer,
)
from nemo_eval.telemetry.metrics import PlanAdherenceScorer
from nemo_eval.telemetry.exporters import TelemetryExporter

__all__ = [
    "HardwareMetrics",
    "HardwareMonitor",
    "ValueExtractor",
    "TrajectoryState",
    "StepEvent",
    "EpisodeTrajectory",
    "TrajectoryTracer",
    "PlanAdherenceScorer",
    "TelemetryExporter",
]
