"""
nemo_eval.cli
--------------
Command-line interface for the Metacognition Evaluation Benchmark Harness.

Usage:
    # Run dual-mode evaluation over MATH benchmark
    python -m nemo_eval.cli run --mode both --dataset math --max-tasks 50

    # Run multi-model evaluation sweep across all 7 target models
    python -m nemo_eval.cli sweep --models all --dataset all --max-tasks 10

    # Run pure zero-shot vanilla evaluation
    python -m nemo_eval.cli run --mode vanilla --dataset putnam --max-tasks 50

    # Run with custom JSON configuration
    python -m nemo_eval.cli run --config config.json

Environment variables (for live remote providers):
    GROQ_API_KEY          — Groq Cloud API key
    OPENAI_API_KEY        — OpenAI-compatible endpoint API key
    NVIDIA_API_KEY        — NVIDIA NIM API key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from nemo_eval.models.registry import TARGET_MODELS, get_model_spec
from nemo_eval.pipeline.config import DatasetSpec, ExecutionMode, ModelSpec, PipelineConfig
from nemo_eval.pipeline.reporter import MarkdownReporter, PipelineReporter
from nemo_eval.pipeline.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemo_eval",
        description="Metacognition Benchmark Evaluation & Dual-Mode Telemetry Harness",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run a benchmark evaluation or dual-mode comparison.")
    _add_runner_arguments(run_p)

    # ── sweep ────────────────────────────────────────────────────────────
    sweep_p = sub.add_parser("sweep", help="Run automated multi-model evaluation sweep.")
    _add_runner_arguments(sweep_p)

    # ── report ───────────────────────────────────────────────────────────
    rep_p = sub.add_parser("report", help="Generate summary scorecards from evaluation output directory.")
    rep_p.add_argument("--output-dir", type=str, default="./results", help="Directory containing run artifacts.")
    rep_p.add_argument("--run-label", type=str, default="Metacognition Benchmark Evaluation")

    # ── audit ────────────────────────────────────────────────────────────
    aud_p = sub.add_parser("audit", help="Run deep latent trace & reasoning span audit over trajectory files.")
    aud_p.add_argument("--dir", type=str, default="./results/live_sweep", help="Directory containing trajectory JSONL files.")
    aud_p.add_argument("--file", type=str, default=None, help="Specific trajectory JSONL file to audit.")
    aud_p.add_argument("--json", action="store_true", help="Output results as structured JSON.")

    # ── validate-config ───────────────────────────────────────────────────
    val_p = sub.add_parser("validate-config", help="Validate a JSON pipeline configuration file.")
    val_p.add_argument("config", type=str, help="Path to config JSON.")

    return parser


def _add_runner_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON PipelineConfig file. Overrides all CLI options.",
    )
    p.add_argument(
        "--mode", type=str, default="both",
        choices=["vanilla", "zero_shot", "agentic", "both", "dual_parity"],
        help="Execution mode: 'vanilla' (0-tool CoT), 'agentic' (9-state FSM), 'both'/'dual_parity' (parity analysis).",
    )
    p.add_argument(
        "--dataset", type=str, default="math",
        help="Dataset to evaluate on: 'math', 'putnam', 'lila', 'all', 'synthetic', 'gsm8k', 'infiagent', 'bird_sql', 'databench'.",
    )
    p.add_argument(
        "--models", type=str, default="mock",
        help="Target models to evaluate. Comma-separated list (e.g. 'Qwen2.5-Math-7B,DeepSeek-R1-7B') or 'all' for all 7 target models.",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Alias for single model provider (mock, groq, openai, vllm, ollama, nemo, nim).",
    )
    p.add_argument(
        "--model-id", type=str, default="",
        help="Provider-specific model ID, e.g. 'deepseek-r1-distill-llama-70b'.",
    )
    p.add_argument(
        "--provider", type=str, default="mock",
        help="Default provider for models (mock, groq, openai, vllm, ollama, nemo, nim).",
    )
    p.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to dataset directory override (optional; defaults to internal deterministic fixtures).",
    )
    p.add_argument(
        "--max-tasks", type=int, default=50,
        help="Limit number of tasks per dataset/category (default: 50).",
    )
    p.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory for scorecards, tables, and JSONL traces (default: ./results).",
    )
    p.add_argument("--max-turns", type=int, default=25, help="Max agent FSM turns before termination.")
    p.add_argument("--max-corrections", type=int, default=3, help="Max self-correction retries per step.")
    p.add_argument("--no-planning", action="store_true", help="Disable DAG task decomposition.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per generation.")
    p.add_argument("--run-label", type=str, default="Metacognition Benchmark Evaluation")


def _build_model_specs(models_arg: str, provider_default: str, model_id_arg: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    if models_arg.lower() == "all":
        model_names = TARGET_MODELS
    else:
        model_names = [m.strip() for m in models_arg.split(",") if m.strip()]

    for m in model_names:
        target_spec = get_model_spec(m)
        provider = provider_default if provider_default != "mock" else target_spec.default_provider
        specs.append(
            ModelSpec(
                name=m,
                provider=provider_default if provider_default in ("mock", "mock_runner") else provider,
                model_id=model_id_arg or m,
            )
        )
    return specs


def cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        mode_val = "both" if args.mode in ("both", "dual_parity") else ("vanilla" if args.mode in ("vanilla", "zero_shot") else "agentic")
        models_input = args.models or args.model or "mock"
        model_specs = _build_model_specs(models_input, args.provider, args.model_id)

        dataset_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
        datasets = [
            DatasetSpec(
                name=d,
                max_tasks=args.max_tasks,
                data_dir=args.data_dir,
            )
            for d in dataset_names
        ]

        config = PipelineConfig(
            run_label=args.run_label,
            output_dir=args.output_dir,
            mode=mode_val,
            max_turns=args.max_turns,
            max_correction_attempts=args.max_corrections,
            enable_planning=not args.no_planning,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            models=model_specs,
            datasets=datasets,
        )

    print(f"\n=======================================================")
    print(f" [CLI] Starting Benchmark Evaluation: {config.run_label}")
    print(f" [CLI] Mode:     {config.mode}")
    print(f" [CLI] Models:   {[m.name for m in config.models]}")
    print(f" [CLI] Datasets: {[d.name for d in config.datasets]}")
    print(f" [CLI] Output:   {config.output_dir}")
    print(f"=======================================================\n")

    runner = BenchmarkRunner(config)
    records = runner.run()

    reporter = MarkdownReporter(output_dir=config.output_dir)
    scorecard_path = reporter.write_summary_report(records, run_label=config.run_label, filename="summary_scorecard.md")
    json_path = reporter.write_json_summary(records, filename="summary.json")
    failure_path = reporter.write_failure_traces(records, filename="failure_traces.md")

    print(f"\n=======================================================")
    print(f" [CLI] Evaluation Complete!")
    print(f"  Scorecard:      {scorecard_path}")
    print(f"  JSON Summary:   {json_path}")
    print(f"  Failure Traces: {failure_path}")
    print(f"=======================================================\n")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        print(f"[CLI] Error: Output directory '{args.output_dir}' does not exist.", file=sys.stderr)
        return 1

    json_summary_file = out_dir / "summary.json"
    if not json_summary_file.exists():
        print(f"[CLI] Warning: '{json_summary_file}' not found. Searching for streaming trajectories...")
        jsonl_files = list(out_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[CLI] Error: No evaluation artifacts found in '{out_dir}'.", file=sys.stderr)
            return 1

    print(f"[CLI] Reports available in {out_dir}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from nemo_eval.eval.trace_audit import TrajectoryTraceAuditor

    auditor = TrajectoryTraceAuditor()
    if args.file:
        res = auditor.audit_file(args.file)
        results = [res] if res else []
    else:
        results = auditor.audit_directory(args.dir)

    if not results:
        print(f"[CLI] No valid trajectory files found to audit.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=========================================================================================")
        print("                         DEEP LATENT TRACE AUDIT SCORECARD                               ")
        print("=========================================================================================")
        print(f"{'Model':<18} {'Dataset':<10} {'Mode':<10} {'Tasks':<6} {'Raw Acc':<10} {'Tool/Latent Acc':<16} {'Timeouts':<10} {'Avg Joules':<12}")
        print("-" * 89)
        for r in results:
            print(
                f"{r['model']:<18} {r['dataset']:<10} {r['mode']:<10} {r['total_tasks']:<6} "
                f"{r['raw_accuracy']:<10} {r['latent_proof_accuracy']:<16} {r['timeouts']:<10} {r['avg_energy_joules']:<12.1f}"
            )
        print("=========================================================================================\n")

    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        config = PipelineConfig.from_json(args.config)
        print(f"[CLI] Config valid: {args.config}")
        print(f"  Mode:     {config.mode}")
        print(f"  Models:   {[m.name for m in config.models]}")
        print(f"  Datasets: {[d.name for d in config.datasets]}")
        return 0
    except Exception as e:
        print(f"[CLI] Config invalid: {e}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("run", "sweep") or (args.command is None and len(sys.argv) > 1):
        if args.command is None:
            # Default to run if arguments given
            args = parser.parse_args(["run"] + sys.argv[1:])
        sys.exit(cmd_run(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))
    elif args.command == "audit":
        sys.exit(cmd_audit(args))
    elif args.command == "validate-config":
        sys.exit(cmd_validate_config(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
