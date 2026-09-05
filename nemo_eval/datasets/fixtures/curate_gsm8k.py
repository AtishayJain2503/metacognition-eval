"""
Curate and generate the 1,000-sample GSM8K benchmark suite:
- Ingest 1,000 test-split grade-school math word problems from OpenAI GSM8K (openai/gsm8k, test split: 1,319 problems).
- Select exactly 1,000 valid tasks deterministically (first 1,000 tasks, index 0 to 999).
- Extract exact integer ground truth from '#### <int>'.
- Normalize ground truth to '\\boxed{<int>}'.
- Standardize Task ID format: 'gsm8k_test_<index:04d>' (gsm8k_test_0000 to gsm8k_test_0999).
- Set benchmark_name: 'gsm8k'.
- Set eval_type: 'float_tol'.
- Populate metadata with split: 'test', reasoning_steps, integer_target, boxed_solution, etc.
- Output offline JSONL fixture: nemo_eval/datasets/fixtures/gsm8k_1000.jsonl (exactly 1,000 lines).
- Output comprehensive CSV catalog: results/gsm8k_tasks_catalog.csv (task_index,task_id,category,level,eval_type,ground_truth,query).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from datasets import load_dataset
from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.engine import evaluate_task_result


def extract_ground_truth(solution_text: str) -> int:
    """Extract integer ground truth from '#### <int>' annotation in GSM8K solution."""
    match = re.search(r"####\s*([\-0-9,]+)", solution_text)
    if not match:
        raise ValueError(f"Could not extract integer answer from solution text: {solution_text!r}")
    raw_str = match.group(1).replace(",", "").strip()
    return int(raw_str)


def count_reasoning_steps(solution_text: str) -> int:
    """
    Compute reasoning steps from the solution text.
    Counts calculation annotations '<<...>>' or non-empty non-annotation lines.
    """
    calc_steps = len(re.findall(r"<<.*?>>", solution_text))
    lines = [
        line.strip()
        for line in solution_text.split("\n")
        if line.strip() and not line.strip().startswith("####")
    ]
    return max(calc_steps, len(lines), 1)


def curate_gsm8k_suite(target_count: int = 1000) -> List[BenchmarkTask]:
    """Ingest and curate exactly target_count tasks from openai/gsm8k test split."""
    print(f"Loading openai/gsm8k (test split)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    tasks: List[BenchmarkTask] = []
    skipped = 0

    for i, row in enumerate(ds):
        if len(tasks) >= target_count:
            break

        question = row["question"].strip()
        answer_text = row["answer"].strip()

        try:
            gt_int = extract_ground_truth(answer_text)
        except Exception as exc:
            print(f"Skipping malformed row {i}: {exc}")
            skipped += 1
            continue

        task_id = f"gsm8k_test_{i:04d}"
        steps = count_reasoning_steps(answer_text)

        query = (
            f"{question}\n\n"
            "Solve this step-by-step using Python code. "
            "Your final answer must be an integer."
        )

        task = BenchmarkTask(
            task_id=task_id,
            benchmark_name="gsm8k",
            query=query,
            context_schema=None,
            db_path=None,
            table_path=None,
            ground_truth=f"\\boxed{{{gt_int}}}",
            eval_type="float_tol",
            metadata={
                "source": "openai/gsm8k",
                "split": "test",
                "index": i,
                "task_index": len(tasks) + 1,
                "category": "Grade School Math",
                "subdiscipline": "Grade School Math",
                "level": "Grade School",
                "reasoning_steps": steps,
                "integer_target": gt_int,
                "boxed_solution": str(gt_int),
                "tolerance": 0.5,
                "original_answer": answer_text,
                "question": question,
            },
        )

        tasks.append(task)

    print(f"Curated {len(tasks)} tasks (skipped {skipped} malformed).")
    if len(tasks) != target_count:
        raise ValueError(f"Expected exactly {target_count} tasks, got {len(tasks)}")

    return tasks


def write_jsonl_fixture(tasks: List[BenchmarkTask], output_path: Path) -> None:
    """Write tasks to JSONL fixture file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")
    print(f"Wrote {len(tasks)} lines to {output_path}")


def write_csv_catalog(tasks: List[BenchmarkTask], output_path: Path) -> None:
    """Export tasks to CSV catalog with exact required columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_index",
        "task_id",
        "category",
        "level",
        "eval_type",
        "ground_truth",
        "query",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, t in enumerate(tasks):
            task_idx = i + 1
            cat = t.metadata.get("category", "Grade School Math")
            lvl = t.metadata.get("level", "Grade School")
            # Flatten query newlines to spaces for clean single-line CSV rows
            clean_query = t.query.replace("\n", " ")
            writer.writerow([
                task_idx,
                t.task_id,
                cat,
                lvl,
                t.eval_type,
                t.ground_truth,
                clean_query,
            ])
    print(f"Exported CSV catalog with {len(tasks)} tasks to {output_path}")


def verify_suite(jsonl_path: Path, csv_path: Path) -> None:
    """Verify generated JSONL fixture and CSV catalog."""
    print(f"Verifying {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 1000, f"Expected 1000 JSONL lines, found {len(lines)}"

    task_ids = set()
    for idx, line in enumerate(lines):
        d = json.loads(line)
        task = BenchmarkTask.model_validate(d)

        assert task.task_id not in task_ids, f"Duplicate task_id {task.task_id}"
        task_ids.add(task.task_id)

        assert task.benchmark_name == "gsm8k", f"Invalid benchmark_name {task.benchmark_name} at {idx}"
        assert task.eval_type == "float_tol", f"Invalid eval_type {task.eval_type} at {idx}"
        assert task.ground_truth.startswith("\\boxed{") and task.ground_truth.endswith("}"), (
            f"Ground truth not boxed: {task.ground_truth} at {idx}"
        )
        assert len(task.query.strip()) > 0, f"Empty query at {idx}"
        assert task.metadata.get("split") == "test", f"Missing split at {idx}"
        assert "reasoning_steps" in task.metadata, f"Missing reasoning_steps at {idx}"
        assert isinstance(task.metadata["reasoning_steps"], int), f"reasoning_steps not int at {idx}"

        # Self-consistency evaluation
        eval_res = evaluate_task_result(task, task.ground_truth)
        assert eval_res.is_correct, (
            f"Self-eval failed for {task.task_id}: {eval_res.diagnostic_message}"
        )

    print(f"JSONL verification passed: 1,000 tasks verified, 100% self-eval pass rate.")

    print(f"Verifying {csv_path}...")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_header = ["task_index", "task_id", "category", "level", "eval_type", "ground_truth", "query"]
        assert header == expected_header, f"Header mismatch: {header} vs {expected_header}"
        rows = list(reader)
        assert len(rows) == 1000, f"Expected 1000 CSV rows, found {len(rows)}"

        for i, row in enumerate(rows):
            assert len(row) == 7, f"Row {i} has {len(row)} columns, expected 7"
            assert int(row[0]) == i + 1, f"Task index mismatch at row {i}: {row[0]}"
            assert row[1] == f"gsm8k_test_{i:04d}", f"Task ID mismatch at row {i}: {row[1]}"
            assert len(row[2].strip()) > 0, f"Empty category at row {i}"
            assert len(row[3].strip()) > 0, f"Empty level at row {i}"
            assert row[4] == "float_tol", f"Invalid eval_type at row {i}: {row[4]}"
            assert row[5].startswith("\\boxed{"), f"Ground truth not boxed at row {i}: {row[5]}"
            assert len(row[6].strip()) > 0, f"Empty query at row {i}"

    print(f"CSV catalog verification passed: 1,000 rows verified with all required columns.")


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    fixtures_dir = repo_root / "nemo_eval" / "datasets" / "fixtures"
    results_dir = repo_root / "results"
    catalogs_dir = results_dir / "catalogs"

    jsonl_output = fixtures_dir / "gsm8k_1000.jsonl"
    csv_output = results_dir / "gsm8k_tasks_catalog.csv"
    alt_csv_output = catalogs_dir / "gsm8k_1000_catalog.csv"

    tasks = curate_gsm8k_suite(target_count=1000)
    write_jsonl_fixture(tasks, jsonl_output)
    write_csv_catalog(tasks, csv_output)
    write_csv_catalog(tasks, alt_csv_output)

    verify_suite(jsonl_output, csv_output)
    print("GSM8K 1,000-sample curation and verification complete!")


if __name__ == "__main__":
    main()
