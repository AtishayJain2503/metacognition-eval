"""
tests.unit.test_datasets.test_math
==================================
Unit tests for MATH benchmark dataset loader (MATHLoader).
"""

import pytest

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.math import MATHLoader


class TestMATHLoader:
    """Test suite for MATHLoader dataset ingestion and contracts."""

    def test_math_loader_instantiation_default(self):
        loader = MATHLoader()
        assert loader.split == TaskSplit.TEST
        assert loader.max_tasks == 50
        assert loader.subject is None

    def test_math_loader_loads_exact_50_tasks(self):
        loader = MATHLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 50
        assert all(isinstance(t, BenchmarkTask) for t in tasks)

    def test_math_loader_subject_coverage(self):
        loader = MATHLoader()
        tasks = loader.load_tasks()
        subjects_found = {t.subdiscipline for t in tasks}
        expected_subjects = {
            "Algebra",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Number Theory",
            "Prealgebra",
            "Precalculus",
        }
        assert expected_subjects.issubset(subjects_found)

    def test_math_loader_subject_filtering(self):
        loader = MATHLoader(subject="Algebra")
        tasks = loader.load_tasks()
        assert len(tasks) > 0
        assert all(t.subdiscipline == "Algebra" for t in tasks)

    def test_math_loader_task_structure_and_types(self):
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=10)
        for t in tasks:
            assert t.task_id.startswith("math_")
            assert t.benchmark_name == "math"
            assert t.dataset_name == "math"
            assert len(t.query) > 10
            assert t.problem_text == t.query
            assert t.ground_truth is not None
            assert t.eval_type in ("math_symbolic", "fraction", "float_tol", "exact")
            assert "level" in t.metadata
            assert "subject" in t.metadata

    def test_math_loader_property_aliases(self):
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=1)
        task = tasks[0]
        assert task.dataset_name == task.benchmark_name
        assert task.problem_text == task.query
        assert task.subdiscipline == task.metadata.get("subject")

    @pytest.mark.parametrize("limit,expected_count", [
        (0, 0),
        (-5, 0),
        (1, 1),
        (10, 10),
        (50, 50),
        (100, 50),  # clamped to available 50
    ])
    def test_math_loader_limit_handling(self, limit, expected_count):
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_count

    def test_math_loader_get_task_by_id(self):
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=5)
        target_id = tasks[0].task_id
        retrieved = loader.get_task(target_id)
        assert retrieved.task_id == target_id
        assert retrieved.query == tasks[0].query

    def test_math_loader_get_task_invalid_id_raises_key_error(self):
        loader = MATHLoader()
        with pytest.raises(KeyError, match="not found in MATH dataset"):
            loader.get_task("non_existent_math_task_9999")

    def test_math_loader_get_subjects(self):
        loader = MATHLoader()
        subjects = loader.get_subjects()
        assert len(subjects) == 7
        assert "Algebra" in subjects
        assert "Precalculus" in subjects

    def test_math_loader_get_manifest(self):
        loader = MATHLoader()
        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "math"
        assert manifest["total_tasks"] == 50
        assert len(manifest["subjects"]) == 7
        assert manifest["split"] == "test"
        assert "offline_fixture" in manifest

    def test_math_loader_load_convenience_alias(self):
        loader = MATHLoader()
        tasks = loader.load(split="test", limit=5, subject="Geometry")
        assert len(tasks) <= 5
        assert all(t.subdiscipline == "Geometry" for t in tasks)

    def test_math_loader_invalid_split_raises_value_error(self):
        loader = MATHLoader()
        with pytest.raises(ValueError, match="Unknown split"):
            loader.load(split="invalid_split_name")

    def test_math_loader_task_serialization_roundtrip(self):
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=1)
        task = tasks[0]
        task_dict = task.to_dict()
        reconstructed = BenchmarkTask.from_dict(task_dict)
        assert reconstructed.task_id == task.task_id
        assert reconstructed.benchmark_name == task.benchmark_name
        assert reconstructed.ground_truth == task.ground_truth

    def test_math_loader_use_1000_fallback_when_absent(self):
        """When use_1000 is requested but math_1000.jsonl is not in default dir, cleanly fallback."""
        loader = MATHLoader(use_1000=True)
        tasks = loader.load_tasks()
        # Either loads math_1000.jsonl (if created) or falls back to 50 tasks
        assert len(tasks) in (50, 1000)

    def test_math_loader_1000_tasks_from_custom_root(self, tmp_path):
        """MATHLoader correctly loads from math_1000.jsonl when present."""
        import json
        custom_dir = tmp_path / "math_data"
        custom_dir.mkdir(parents=True, exist_ok=True)
        sample_tasks = [
            {
                "task_id": f"math_alg_{i:04d}",
                "benchmark_name": "math",
                "query": f"Problem {i}. Final answer in \\boxed{{}}.",
                "ground_truth": f"\\boxed{{{i}}}",
                "eval_type": "math_symbolic",
                "metadata": {"subject": "Algebra", "level": 1},
            }
            for i in range(1, 101)
        ]
        with open(custom_dir / "math_1000.jsonl", "w", encoding="utf-8") as f:
            for t in sample_tasks:
                f.write(json.dumps(t) + "\n")

        loader = MATHLoader(dataset_root=str(custom_dir), use_1000=True, max_tasks=100)
        tasks = loader.load_tasks()
        assert len(tasks) == 100
        assert tasks[0].task_id == "math_alg_0001"
