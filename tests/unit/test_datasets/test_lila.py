"""
tests.unit.test_datasets.test_lila
==================================
Unit tests for AllenAI Lila benchmark dataset loader (LilaLoader).
"""

from collections import Counter
import pytest

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.lila import LilaLoader


class TestLilaLoader:
    """Test suite for LilaLoader multi-domain benchmark ingestion."""

    def test_lila_loader_instantiation_default(self):
        loader = LilaLoader()
        assert loader.split == TaskSplit.TEST
        assert len(loader.subcategories) == 7
        assert loader.max_tasks_per_category == 50

    def test_lila_loader_loads_all_350_tasks(self):
        loader = LilaLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 350
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_lila_loader_7_subcategories_distribution(self):
        loader = LilaLoader()
        tasks = loader.load_tasks()
        counts = Counter(t.subdiscipline.lower() for t in tasks)
        expected_subcats = [
            "arithmetic",
            "algebra",
            "calculus",
            "geometry",
            "combinatorics",
            "physics",
            "statistics",
        ]
        for subcat in expected_subcats:
            assert counts[subcat] == 50

    def test_lila_loader_load_category(self):
        loader = LilaLoader()
        calculus_tasks = loader.load_category("calculus", limit=50)
        assert len(calculus_tasks) == 50
        assert all(t.subdiscipline.lower() == "calculus" for t in calculus_tasks)

    def test_lila_loader_subset_subcategories_init(self):
        loader = LilaLoader(subcategories=["arithmetic", "algebra"])
        tasks = loader.load_tasks()
        assert len(tasks) == 100
        found_subcats = {t.subdiscipline.lower() for t in tasks}
        assert found_subcats == {"arithmetic", "algebra"}

    def test_lila_loader_polymorphic_eval_types(self):
        loader = LilaLoader()
        tasks = loader.load_tasks()
        eval_types = {t.eval_type for t in tasks}
        assert "exact" in eval_types
        assert "math_symbolic" in eval_types
        assert "float_tol" in eval_types
        assert "set" in eval_types

    def test_lila_loader_property_aliases(self):
        loader = LilaLoader()
        tasks = loader.load_tasks(limit=1)
        task = tasks[0]
        assert task.dataset_name == "lila"
        assert task.problem_text == task.query
        assert task.subdiscipline in ("Arithmetic", "arithmetic")

    @pytest.mark.parametrize("limit,expected_count", [
        (0, 0),
        (-1, 0),
        (1, 1),
        (25, 25),
        (100, 100),
        (350, 350),
        (500, 350),  # clamped to available 350
    ])
    def test_lila_loader_limit_handling(self, limit, expected_count):
        loader = LilaLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_count

    def test_lila_loader_get_task_by_id(self):
        loader = LilaLoader()
        tasks = loader.load_tasks(limit=5)
        target_id = tasks[0].task_id
        retrieved = loader.get_task(target_id)
        assert retrieved.task_id == target_id
        assert retrieved.query == tasks[0].query

    def test_lila_loader_get_task_invalid_id_raises_key_error(self):
        loader = LilaLoader()
        with pytest.raises(KeyError, match="not found in Lila dataset"):
            loader.get_task("invalid_lila_task_id_8888")

    def test_lila_loader_get_subcategories(self):
        loader = LilaLoader()
        subcategories = loader.get_subcategories()
        assert len(subcategories) == 7
        assert "arithmetic" in subcategories
        assert "statistics" in subcategories

    def test_lila_loader_get_manifest(self):
        loader = LilaLoader()
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "lila"
        assert manifest["total_tasks"] == 350
        assert len(manifest["subcategories"]) == 7
        assert manifest["split"] == "test"
        assert "offline_fixture" in manifest

    def test_lila_loader_load_convenience_alias(self):
        loader = LilaLoader()
        tasks = loader.load(split="test", limit=10, subdiscipline="Physics")
        assert len(tasks) == 10
        assert all(t.subdiscipline.lower() == "physics" for t in tasks)

    def test_lila_loader_invalid_split_raises_value_error(self):
        loader = LilaLoader()
        with pytest.raises(ValueError, match="Unknown split"):
            loader.load(split="invalid_split_name_bar")
