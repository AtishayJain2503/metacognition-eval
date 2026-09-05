"""
tests.unit.test_datasets.test_putnam
====================================
Unit tests for PutnamBench benchmark dataset loader (PutnamBenchLoader).
"""

import pytest

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.putnam import PutnamBenchLoader


class TestPutnamBenchLoader:
    """Test suite for PutnamBenchLoader competition benchmark ingestion."""

    def test_putnam_loader_instantiation_default(self):
        loader = PutnamBenchLoader()
        assert loader.split == TaskSplit.TEST
        assert loader.max_tasks == 50
        assert loader.category is None

    def test_putnam_loader_loads_exact_50_tasks(self):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 50
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_putnam_loader_categories_coverage(self):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks()
        cats_found = {t.subdiscipline.lower().replace(" ", "_") for t in tasks}
        expected_cats = {
            "real_analysis",
            "abstract_algebra",
            "linear_algebra",
            "number_theory",
            "combinatorics",
            "geometry",
            "calculus",
        }
        assert expected_cats.issubset(cats_found)

    def test_putnam_loader_category_filtering(self):
        loader = PutnamBenchLoader(category="real_analysis")
        tasks = loader.load_tasks()
        assert len(tasks) > 0
        assert all(t.subdiscipline.lower().replace(" ", "_") == "real_analysis" for t in tasks)

    def test_putnam_loader_task_contract(self):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks(limit=10)
        for t in tasks:
            assert t.task_id.startswith("putnam_")
            assert t.benchmark_name == "putnam"
            assert t.dataset_name == "putnam"
            assert "Putnam" in t.query
            assert t.problem_text == t.query
            assert t.ground_truth is not None
            assert t.eval_type in ("math_symbolic", "fraction", "exact", "set")
            assert t.metadata.get("formal_verification") is True
            assert "year" in t.metadata

    def test_putnam_loader_property_aliases(self):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks(limit=1)
        task = tasks[0]
        assert task.dataset_name == "putnam"
        assert task.problem_text == task.query
        assert task.subdiscipline == task.metadata.get("category")

    @pytest.mark.parametrize("limit,expected_count", [
        (0, 0),
        (-1, 0),
        (1, 1),
        (20, 20),
        (50, 50),
        (100, 50),
    ])
    def test_putnam_loader_limit_handling(self, limit, expected_count):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_count

    def test_putnam_loader_get_task_by_id(self):
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks(limit=5)
        target_id = tasks[0].task_id
        retrieved = loader.get_task(target_id)
        assert retrieved.task_id == target_id
        assert retrieved.query == tasks[0].query

    def test_putnam_loader_get_task_invalid_id_raises_key_error(self):
        loader = PutnamBenchLoader()
        with pytest.raises(KeyError, match="not found in PutnamBench dataset"):
            loader.get_task("invalid_putnam_task_id_999")

    def test_putnam_loader_get_categories(self):
        loader = PutnamBenchLoader()
        categories = loader.get_categories()
        assert len(categories) == 7
        assert "real_analysis" in categories
        assert "combinatorics" in categories

    def test_putnam_loader_get_manifest(self):
        loader = PutnamBenchLoader()
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "putnam"
        assert manifest["total_tasks"] == 50
        assert len(manifest["categories"]) == 7
        assert manifest["split"] == "test"
        assert "offline_fixture" in manifest

    def test_putnam_loader_load_convenience_alias(self):
        loader = PutnamBenchLoader()
        tasks = loader.load(split="test", limit=5, category="calculus")
        assert len(tasks) <= 5
        assert all(t.subdiscipline.lower() == "calculus" for t in tasks)

    def test_putnam_loader_invalid_split_raises_value_error(self):
        loader = PutnamBenchLoader()
        with pytest.raises(ValueError, match="Unknown split"):
            loader.load(split="invalid_split_foo")

    def test_putnam_loader_use_1000_fallback_when_absent(self):
        """When use_1000 is requested but putnam_1000.jsonl is absent, cleanly fallback."""
        loader = PutnamBenchLoader(use_1000=True)
        tasks = loader.load_tasks()
        assert len(tasks) in (50, 1000)

    def test_putnam_loader_1000_tasks_from_custom_root(self, tmp_path):
        """PutnamBenchLoader correctly loads from putnam_1000.jsonl when present."""
        import json
        custom_dir = tmp_path / "putnam_data"
        custom_dir.mkdir(parents=True, exist_ok=True)
        sample_tasks = [
            {
                "task_id": f"putnam_comp_{i:04d}",
                "benchmark_name": "putnam",
                "query": f"Putnam Problem {i}. Final answer in \\boxed{{}}.",
                "ground_truth": f"\\boxed{{{i}}}",
                "eval_type": "math_symbolic",
                "metadata": {"category": "calculus", "year": 2020},
            }
            for i in range(1, 101)
        ]
        with open(custom_dir / "putnam_1000.jsonl", "w", encoding="utf-8") as f:
            for t in sample_tasks:
                f.write(json.dumps(t) + "\n")

        loader = PutnamBenchLoader(dataset_root=str(custom_dir), use_1000=True, max_tasks=100)
        tasks = loader.load_tasks()
        assert len(tasks) == 100
        assert tasks[0].task_id == "putnam_comp_0001"
