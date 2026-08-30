"""
nemo_eval.cli
--------------
Command-line interface for the NeMo Long-Horizon Agent Evaluation Harness.

Usage:
    python -m nemo_eval.cli run --config config.json
    python -m nemo_eval.cli run --model mock --dataset synthetic --max-tasks 10
    python -m nemo_eval.cli run --model groq --model-id llama-3.3-70b-versatile --dataset synthetic

Environment variables (for real providers):
    GROQ_API_KEY          — Groq Cloud API key
    OPENAI_API_KEY        — OpenAI-compatible endpoint API key
    NVIDIA_API_KEY        — NVIDIA NIM API key
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemo_eval.pipeline.config import PipelineConfig, ModelSpec, DatasetSpec
from nemo_eval.pipeline.runner import BenchmarkRunner
from nemo_eval.pipeline.reporter import PipelineReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemo_eval",
        description="NeMo Long-Horizon Agent Evaluation Harness",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run an evaluation sweep.")
    run_p.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON PipelineConfig file. Overrides all other options.",
    )
    run_p.add_argument("--model", type=str, default="mock",
        choices=["mock", "groq", "openai", "openai_gateway", "nemo", "nim"],
        help="Model provider (default: mock).",
    )
    run_p.add_argument("--model-id", type=str, default="",
        help="Provider-specific model ID, e.g. 'llama-3.3-70b-versatile'.",
    )
    run_p.add_argument("--dataset", type=str, default="synthetic",
        choices=["synthetic", "infiagent", "bird_sql", "databench"],
        help="Dataset to evaluate on (default: synthetic).",
    )
    run_p.add_argument("--data-dir", type=str, default=None,
        help="Path to dataset files (required for non-synthetic datasets).",
    )
    run_p.add_argument("--max-tasks", type=int, default=None,
        help="Limit number of tasks (default: all).",
    )
    run_p.add_argument("--output-dir", type=str, default="./nemo_eval_output",
        help="Output directory for run artifacts.",
    )
    run_p.add_argument("--max-turns", type=int, default=25)
    run_p.add_argument("--max-corrections", type=int, default=3)
    run_p.add_argument("--no-planning", action="store_true",
        help="Disable task decomposition planning (single-step mode).",
    )
    run_p.add_argument("--run-label", type=str, default="nemo_eval_run")

    # ── validate-config ───────────────────────────────────────────────────
    val_p = sub.add_parser("validate-config", help="Validate a JSON config file.")
    val_p.add_argument("config", type=str, help="Path to config JSON.")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig(
            run_label=args.run_label,
            output_dir=args.output_dir,
            max_turns=args.max_turns,
            max_correction_attempts=args.max_corrections,
            enable_planning=not args.no_planning,
            models=[
                ModelSpec(
                    name=f"{args.model}/{args.model_id or 'default'}",
                    provider=args.model,
                    model_id=args.model_id or "",
                )
            ],
            datasets=[
                DatasetSpec(
                    name=args.dataset,
                    max_tasks=args.max_tasks,
                    data_dir=args.data_dir,
                )
            ],
        )

    print(f"[CLI] Starting evaluation: {config.run_label}")
    print(f"[CLI] Models: {[m.name for m in config.models]}")
    print(f"[CLI] Datasets: {[d.name for d in config.datasets]}")
    print(f"[CLI] Output: {config.output_dir}")
    print()

    runner = BenchmarkRunner(config)
    records = runner.run()

    reporter = PipelineReporter(output_dir=config.output_dir)
    report_path = reporter.write_summary_report(records, run_label=config.run_label)
    json_path = reporter.write_json_summary(records)
    failure_path = reporter.write_failure_traces(records)

    print(f"\n[CLI] Evaluation complete.")
    print(f"  Summary: {report_path}")
    print(f"  JSON:    {json_path}")
    print(f"  Failures:{failure_path}")

    # Exit code 0 if any model succeeded, 1 if all failed
    any_success = any(
        any(t.status == "success" for t in r.trajectories)
        for r in records
    )
    return 0 if any_success or not records else 1


def cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        config = PipelineConfig.from_json(args.config)
        print(f"[CLI] Config valid: {args.config}")
        print(f"  Models: {[m.name for m in config.models]}")
        print(f"  Datasets: {[d.name for d in config.datasets]}")
        return 0
    except Exception as e:
        print(f"[CLI] Config invalid: {e}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "validate-config":
        sys.exit(cmd_validate_config(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
