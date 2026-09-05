"""
tests.unit.test_datasets.test_svamp
===================================
Unit tests for SVAMP benchmark dataset loader (SVAMPLoader).
"""

import json
import pytest
from pathlib import Path

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.svamp import SVAMPLoader


@pytest.fixture
def temp_svamp_fixture(tmp_path):
    """Create a mock svamp_1000.jsonl fixture with sample tasks."""
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "svamp_1000.jsonl"

    sample_tasks = [
        {
            "task_id": f"svamp_chal_{i}",
            "benchmark_name": "svamp",
            "query": f"Word problem variation {i}. Put answer in \\boxed{{}}.",
            "ground_truth": f"\\boxed{{{i * 2}}}",
            "eval_type": "float_tol",
            "metadata": {
                "source": "svamp",
                "type": "Common-Addition" if i % 2 == 0 else "Common-Subtraction",
                "category": "Common-Addition" if i % 2 == 0 else "Common-Subtraction",
                "split": "test",
                "index": i,
            },
        }
        for i in range(1, 21)
    ]

    with open(fixture_path, "w", encoding="utf-8") as f:
        for t in sample_tasks:
            f.write(json.dumps(t) + "\n")

    return fixture_dir


class TestSVAMPLoader:
    """Test suite for SVAMPLoader dataset ingestion and contracts."""

    def test_svamp_loader_instantiation_default(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        assert loader.split == TaskSplit.TEST
        assert loader.max_tasks is None
        assert loader.category is None
        assert loader.use_fixture is True

    def test_svamp_loader_loads_tasks(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        tasks = loader.load_tasks()
        assert len(tasks) == 20
        assert all(isinstance(t, BenchmarkTask) for t in tasks)
        assert all(t.benchmark_name == "svamp" for t in tasks)
        assert all(t.eval_type == "float_tol" for t in tasks)

    def test_svamp_loader_category_filtering(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture), category="Common-Addition")
        tasks = loader.load_tasks()
        assert len(tasks) == 10
        assert all(t.metadata.get("type") == "Common-Addition" for t in tasks)

    def test_svamp_loader_category_filtering_case_insensitive(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture), category="  common_addition  ")
        tasks = loader.load_tasks()
        assert len(tasks) == 10

    @pytest.mark.parametrize("limit,expected_count", [
        (0, 0),
        (-5, 0),
        (1, 1),
        (5, 5),
        (20, 20),
        (50, 20),  # clamped to available 20
    ])
    def test_svamp_loader_limit_handling(self, temp_svamp_fixture, limit, expected_count):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_count

    def test_svamp_loader_get_task_by_id(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        task = loader.get_task("svamp_chal_1")
        assert task.task_id == "svamp_chal_1"
        assert task.benchmark_name == "svamp"

    def test_svamp_loader_get_task_invalid_id_raises_key_error(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        with pytest.raises(KeyError, match="not found in SVAMP dataset"):
            loader.get_task("svamp_nonexistent_9999")

    def test_svamp_loader_get_types(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        types = loader.get_types()
        assert len(types) == 4
        assert "Common-Addition" in types
        assert "Common-Division" in types

    def test_svamp_loader_get_manifest(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "svamp"
        assert manifest["total_tasks"] == 20
        assert manifest["split"] == "test"
        assert len(manifest["types"]) == 4

    def test_svamp_loader_load_convenience_alias(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        tasks = loader.load(split="test", limit=3, category="Common-Subtraction")
        assert len(tasks) == 3
        assert all(t.metadata.get("type") == "Common-Subtraction" for t in tasks)

    def test_svamp_loader_invalid_split_raises_value_error(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        with pytest.raises(ValueError, match="Unknown split"):
            loader.load(split="invalid_split_name")

    def test_svamp_loader_task_serialization_roundtrip(self, temp_svamp_fixture):
        loader = SVAMPLoader(dataset_root=str(temp_svamp_fixture))
        task = loader.get_task("svamp_chal_1")
        d = task.to_dict()
        reconstructed = BenchmarkTask.from_dict(d)
        assert reconstructed.task_id == task.task_id
        assert reconstructed.benchmark_name == task.benchmark_name
        assert reconstructed.ground_truth == task.ground_truth
