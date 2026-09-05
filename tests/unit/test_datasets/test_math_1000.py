"""
tests.unit.test_datasets.test_math_1000
=======================================
Unit tests verifying the curated 1,000-sample Hendrycks MATH benchmark suite.
Covers schema compliance, stratification distribution, CSV catalog consistency,
and loader integration.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.eval.math_eval import evaluate_math_expression, extract_latex_boxed


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES_DIR = PROJECT_ROOT / "nemo_eval" / "datasets" / "fixtures"
RESULTS_DIR = PROJECT_ROOT / "results"

JSONL_PATH = FIXTURES_DIR / "math_1000.jsonl"
CSV_PATH = RESULTS_DIR / "math_tasks_catalog.csv"

EXPECTED_SUBJECTS = {
    "Algebra": 237,
    "Counting & Probability": 95,
    "Geometry": 96,
    "Intermediate Algebra": 181,
    "Number Theory": 108,
    "Prealgebra": 174,
    "Precalculus": 109,
}


class TestMath1000Suite:
    """Verification suite for the 1,000-sample Hendrycks MATH benchmark."""

    def test_fixture_file_exists(self):
        assert JSONL_PATH.exists(), f"Fixture file missing: {JSONL_PATH}"

    def test_catalog_file_exists(self):
        assert CSV_PATH.exists(), f"Catalog file missing: {CSV_PATH}"

    def test_jsonl_exact_1000_lines(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1000

    def test_jsonl_valid_json_and_pydantic_schema(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            data = json.loads(line)
            task = BenchmarkTask.model_validate(data)
            assert task.benchmark_name == "math"
            assert task.eval_type == "math_symbolic"
            assert task.task_id.startswith("math_")
            assert len(task.query) > 10
            assert task.ground_truth.startswith(r"\boxed{")
            assert task.ground_truth.endswith("}")
            assert extract_latex_boxed(task.ground_truth) is not None

    def test_jsonl_task_ids_unique(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        task_ids = [json.loads(line)["task_id"] for line in lines]
        assert len(task_ids) == 1000
        assert len(set(task_ids)) == 1000

    def test_stratified_subject_proportionality(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        subject_counts = Counter()
        for line in lines:
            rec = json.loads(line)
            subj = rec["metadata"]["subject"]
            subject_counts[subj] += 1

        assert sum(subject_counts.values()) == 1000
        for subj, expected_count in EXPECTED_SUBJECTS.items():
            assert subject_counts[subj] == expected_count, (
                f"Subject {subj} count {subject_counts[subj]} != expected {expected_count}"
            )

    def test_difficulty_level_representation(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        level_counts = Counter()
        for line in lines:
            rec = json.loads(line)
            level = rec["metadata"]["level"]
            level_counts[level] += 1

        assert sum(level_counts.values()) == 1000
        assert set(level_counts.keys()) == {1, 2, 3, 4, 5}
        # Verify non-trivial representation in each difficulty tier
        for lvl in range(1, 6):
            assert level_counts[lvl] >= 80

    def test_metadata_fields_populated(self):
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            rec = json.loads(line)
            meta = rec["metadata"]
            assert meta["subject"] in EXPECTED_SUBJECTS
            assert meta["level"] in (1, 2, 3, 4, 5)
            assert meta["split"] == "test"
            assert meta["source"] == "EleutherAI/hendrycks_math"
            assert "original_config" in meta
            assert "original_index" in meta
            assert "boxed_solution" in meta
            assert len(meta["boxed_solution"]) > 0

    def test_csv_catalog_exact_1000_rows(self):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_headers = ["task_index", "task_id", "category", "level", "eval_type", "ground_truth", "query"]
            assert reader.fieldnames == expected_headers
            rows = list(reader)
        assert len(rows) == 1000

    def test_csv_catalog_matches_jsonl_records(self):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        jsonl_records = [
            json.loads(line)
            for line in JSONL_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(csv_rows) == len(jsonl_records) == 1000

        for i in range(1000):
            row = csv_rows[i]
            rec = jsonl_records[i]
            assert int(row["task_index"]) == i + 1
            assert row["task_id"] == rec["task_id"]
            assert row["category"] == rec["metadata"]["subject"]
            assert int(row["level"]) == rec["metadata"]["level"]
            assert row["eval_type"] == rec["eval_type"]
            assert row["ground_truth"] == rec["ground_truth"]
            assert row["query"] == rec["query"]

    def test_math_loader_integration_with_1000_tasks(self):
        loader = MATHLoader(use_1000=True, max_tasks=1000)
        tasks = loader.load_tasks()
        assert len(tasks) == 1000
        assert all(isinstance(t, BenchmarkTask) for t in tasks)
        assert all(t.benchmark_name == "math" for t in tasks)
        assert all(t.eval_type == "math_symbolic" for t in tasks)

    def test_math_loader_backward_compatibility_preserved(self):
        # Default MATHLoader without arguments loads 50 tasks from math_tasks.jsonl
        loader = MATHLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 50

    def test_symbolic_evaluator_evaluates_curated_ground_truth(self):
        # Verify that sample ground truths can be evaluated correctly
        lines = [line.strip() for line in JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines[:10]:
            rec = json.loads(line)
            gold = rec["ground_truth"]
            eval_res = evaluate_math_expression(candidate=gold, gold=gold, eval_type="math_symbolic")
            assert eval_res.is_correct is True
