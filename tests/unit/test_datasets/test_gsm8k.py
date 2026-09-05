"""
tests.unit.test_datasets.test_gsm8k
===================================
Unit tests for GSM8K benchmark dataset loader (GSM8KLoader).
"""

import json
import pytest
from pathlib import Path

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.gsm8k import GSM8KLoader, _extract_answer


@pytest.fixture
def temp_gsm8k_fixture(tmp_path):
    """Create a mock gsm8k_1000.jsonl fixture with sample tasks."""
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "gsm8k_1000.jsonl"

    sample_tasks = [
        {
            "task_id": f"gsm8k_test_{i:04d}",
            "benchmark_name": "gsm8k",
            "query": f"Word problem {i}. Solve step by step. Final answer in \\boxed{{}}.",
            "ground_truth": f"\\boxed{{{i * 10}}}",
            "eval_type": "float_tol",
            "metadata": {
                "source": "gsm8k",
                "split": "test",
                "index": i,
                "original_answer": f"Explanation steps #### {i * 10}",
            },
        }
        for i in range(1, 21)
    ]

    with open(fixture_path, "w", encoding="utf-8") as f:
        for t in sample_tasks:
            f.write(json.dumps(t) + "\n")

    return fixture_dir


class TestExtractAnswer:
    """Test the _extract_answer helper."""

    def test_extract_positive_integer(self):
        assert _extract_answer("The answer is 42. #### 42") == 42

    def test_extract_with_commas(self):
        assert _extract_answer("Total sum: #### 1,000,000") == 1000000

    def test_extract_negative_integer(self):
        assert _extract_answer("Result: #### -15") == -15

    def test_extract_no_match_returns_none(self):
        assert _extract_answer("No answer tag here") is None

    def test_extract_invalid_number_returns_none(self):
        assert _extract_answer("Result: #### abc") is None


class TestGSM8KLoader:
    """Test suite for GSM8KLoader dataset ingestion and contracts."""

    def test_gsm8k_loader_instantiation_default(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        assert loader.split == TaskSplit.TEST
        assert loader.max_tasks is None
        assert loader.use_fixture is True

    def test_gsm8k_loader_instantiation_positional_split(self, temp_gsm8k_fixture):
        loader = GSM8KLoader("test")
        assert loader.split == TaskSplit.TEST

    def test_gsm8k_loader_loads_tasks(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        tasks = loader.load_tasks()
        assert len(tasks) == 20
        assert all(isinstance(t, BenchmarkTask) for t in tasks)
        assert all(t.benchmark_name == "gsm8k" for t in tasks)
        assert all(t.eval_type == "float_tol" for t in tasks)

    @pytest.mark.parametrize("limit,expected_count", [
        (0, 0),
        (-5, 0),
        (1, 1),
        (5, 5),
        (20, 20),
        (50, 20),  # clamped to available 20
    ])
    def test_gsm8k_loader_limit_handling(self, temp_gsm8k_fixture, limit, expected_count):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_count

    def test_gsm8k_loader_get_task_by_id(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        task = loader.get_task("gsm8k_test_0001")
        assert task.task_id == "gsm8k_test_0001"
        assert task.benchmark_name == "gsm8k"

    def test_gsm8k_loader_get_task_invalid_id_raises_key_error(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        with pytest.raises(KeyError, match="not found in GSM8K dataset"):
            loader.get_task("gsm8k_nonexistent_9999")

    def test_gsm8k_loader_get_manifest(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "gsm8k"
        assert manifest["total_tasks"] == 20
        assert manifest["split"] == "test"
        assert "offline_fixture" in manifest

    def test_gsm8k_loader_load_convenience_alias(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        tasks = loader.load(split="test", limit=3)
        assert len(tasks) == 3

    def test_gsm8k_loader_invalid_split_raises_value_error(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        with pytest.raises(ValueError, match="Unknown split"):
            loader.load(split="invalid_split_name")

    def test_gsm8k_loader_task_serialization_roundtrip(self, temp_gsm8k_fixture):
        loader = GSM8KLoader(dataset_root=str(temp_gsm8k_fixture))
        task = loader.get_task("gsm8k_test_0001")
        d = task.to_dict()
        reconstructed = BenchmarkTask.from_dict(d)
        assert reconstructed.task_id == task.task_id
        assert reconstructed.benchmark_name == task.benchmark_name
        assert reconstructed.ground_truth == task.ground_truth
