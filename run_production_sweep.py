import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_eval.pipeline.config import PipelineConfig
from nemo_eval.pipeline.runner import BenchmarkRunner
from nemo_eval.pipeline.reporter import MarkdownReporter

def main():
    parser = argparse.ArgumentParser(description="Metacognition Benchmark Production Sweep CLI")
    parser.add_argument("--config", type=str, default="configs/production_sweep_config.json", help="Path to config file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks for a quick validation run.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading configuration from {config_path}...")
    config = PipelineConfig.from_json(config_path)

    if args.limit is not None:
        print(f"Applying limit override: max {args.limit} tasks per dataset.")
        for ds in config.datasets:
            ds.max_tasks = args.limit

    # Execute the sweep
    runner = BenchmarkRunner(config)
    print("Initializing benchmark execution sweep...")
    records = runner.run()

    # Generate Markdown Scorecards and JSON Summary
    print("Generating report files...")
    reporter = MarkdownReporter(output_dir=config.output_dir)
    md_path = reporter.write_summary_report(records, run_label=config.run_label, filename="production_scorecard.md")
    json_path = reporter.write_json_summary(records, filename="production_summary.json")
    failure_path = reporter.write_failure_traces(records, filename="production_failures.md")

    print("\n" + "="*80)
    print(" PRODUCTION SWEEP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"  • Summary Scorecard:  {md_path}")
    print(f"  • JSON Summary:       {json_path}")
    print(f"  • Failure Traces:     {failure_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
