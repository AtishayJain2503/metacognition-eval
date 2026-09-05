"""
tests.unit.test_datasets.test_gsm8k_1000
=========================================
Comprehensive verification tests for the 1,000-sample GSM8K benchmark suite:
- Fixture integrity: nemo_eval/datasets/fixtures/gsm8k_1000.jsonl (1,000 tasks)
- CSV catalog: results/gsm8k_tasks_catalog.csv (1,000 tasks)
- Schema conformance and normalization
- Ground-truth boxed format and integer extraction
- Metadata consistency (split, reasoning_steps, integer_target)
- Evaluator compatibility and self-consistency
- Loader integration via GSM8KLoader
"""

import csv
import json
import re
from pathlib import Path
import pytest

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.datasets.gsm8k import GSM8KLoader
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.telemetry.extractor import ValueExtractor


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURE_PATH = REPO_ROOT / "nemo_eval" / "datasets" / "fixtures" / "gsm8k_1000.jsonl"
CATALOG_PATH = REPO_ROOT / "results" / "gsm8k_tasks_catalog.csv"


class TestGSM8K1000Fixture:
    """Verification suite for gsm8k_1000.jsonl fixture."""

    def test_fixture_file_exists(self):
        assert FIXTURE_PATH.is_file(), f"Fixture file not found: {FIXTURE_PATH}"

    def test_exact_1000_records(self):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1000, f"Expected 1000 lines, found {len(lines)}"

    def test_schema_conformance_and_uniqueness(self):
        seen_ids = set()
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                task = BenchmarkTask.model_validate(data)

                # Uniqueness
                assert task.task_id not in seen_ids, f"Duplicate task_id {task.task_id} at line {idx}"
                seen_ids.add(task.task_id)

                # Expected task_id format
                assert re.match(r"^gsm8k_test_\d{4}$", task.task_id), (
                    f"Task ID format mismatch: {task.task_id}"
                )

                # Contract fields
                assert task.benchmark_name == "gsm8k"
                assert task.eval_type == "float_tol"
                assert len(task.query.strip()) > 0

                # Ground truth boxed integer format
                gt = task.ground_truth
                assert gt.startswith("\\boxed{") and gt.endswith("}"), f"Ground truth not boxed: {gt}"
                inner = gt[7:-1].strip()
                assert re.match(r"^-?\d+$", inner), f"Inner boxed value is not an integer: {inner}"

                # Metadata fields
                meta = task.metadata
                assert meta.get("split") == "test"
                assert "reasoning_steps" in meta
                assert isinstance(meta["reasoning_steps"], int) and meta["reasoning_steps"] >= 1
                assert "integer_target" in meta
                assert int(meta["integer_target"]) == int(inner)
                assert meta.get("source") == "openai/gsm8k"

        assert len(seen_ids) == 1000

    def test_evaluator_compatibility_sample(self):
        """Verify ground-truth self-evaluation across representative sample."""
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Test every 20th task (50 tasks across the full distribution)
        sample_indices = range(0, 1000, 20)
        for idx in sample_indices:
            task = BenchmarkTask.model_validate(json.loads(lines[idx]))
            eval_res = evaluate_task_result(task, task.ground_truth)
            assert eval_res.is_correct is True, (
                f"Self-eval failed for task {task.task_id}: {eval_res.diagnostic_message}"
            )
            assert eval_res.score == 1.0

            # ValueExtractor compatibility
            extracted = ValueExtractor.extract_value(task.ground_truth, expected_type="float_tol")
            expected_num = str(task.metadata["integer_target"])
            assert extracted == expected_num, (
                f"ValueExtractor mismatch for {task.task_id}: extracted={extracted}, expected={expected_num}"
            )


class TestGSM8KTasksCatalogCSV:
    """Verification suite for gsm8k_tasks_catalog.csv."""

    def test_catalog_file_exists(self):
        assert CATALOG_PATH.is_file(), f"Catalog file not found: {CATALOG_PATH}"

    def test_catalog_header_and_row_count(self):
        with open(CATALOG_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = [
                "task_index",
                "task_id",
                "category",
                "level",
                "eval_type",
                "ground_truth",
                "query",
            ]
            assert header == expected_header, f"Header mismatch: {header}"

            rows = list(reader)
            assert len(rows) == 1000, f"Expected 1000 data rows, found {len(rows)}"

    def test_catalog_row_consistency_with_fixture(self):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as fj, \
             open(CATALOG_PATH, "r", encoding="utf-8", newline="") as fc:
            json_lines = [json.loads(l) for l in fj if l.strip()]
            reader = csv.reader(fc)
            next(reader)  # skip header
            csv_rows = list(reader)

            for i in range(1000):
                jt = json_lines[i]
                cr = csv_rows[i]

                # task_index
                assert int(cr[0]) == i + 1
                # task_id
                assert cr[1] == jt["task_id"]
                # category
                assert cr[2] == jt["metadata"]["category"]
                # level
                assert cr[3] == jt["metadata"]["level"]
                # eval_type
                assert cr[4] == jt["eval_type"]
                # ground_truth
                assert cr[5] == jt["ground_truth"]
                # query (flattened)
                assert len(cr[6].strip()) > 0


class TestGSM8KLoaderIntegration:
    """Integration test suite for GSM8KLoader with gsm8k_1000.jsonl fixture."""

    def test_loader_loads_default_1000_tasks(self):
        loader = GSM8KLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 1000
        assert tasks[0].task_id == "gsm8k_test_0000"
        assert tasks[-1].task_id == "gsm8k_test_0999"

    def test_loader_limit_handling(self):
        loader = GSM8KLoader()
        assert len(loader.load_tasks(limit=5)) == 5
        assert len(loader.load_tasks(limit=100)) == 100
        assert len(loader.load_tasks(limit=1500)) == 1000  # Clamped to available 1000

    def test_loader_get_task_boundary(self):
        loader = GSM8KLoader()
        first_task = loader.get_task("gsm8k_test_0000")
        assert first_task.task_id == "gsm8k_test_0000"
        assert first_task.ground_truth == "\\boxed{18}"

        last_task = loader.get_task("gsm8k_test_0999")
        assert last_task.task_id == "gsm8k_test_0999"
        assert last_task.ground_truth == "\\boxed{25}"

    def test_loader_manifest(self):
        loader = GSM8KLoader()
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "gsm8k"
        assert manifest["total_tasks"] == 1000
        assert manifest["split"] == "test"
