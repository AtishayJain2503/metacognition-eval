"""
tests.unit.test_datasets.test_base
==================================
Unit tests for BenchmarkTask, TaskSplit, and BaseDatasetLoader ABC contracts.
"""

from typing import Any, Dict, List, Optional
import pytest
from pydantic import ValidationError

from nemo_eval.datasets.base import (
    BaseDatasetLoader,
    BenchmarkTask,
    TaskSplit,
)


class TestBenchmarkTaskModel:
    """Test validation, serialization, and deserialization of BenchmarkTask."""

    def test_benchmark_task_instantiation_valid(self):
        """Task instantiates with all valid parameters."""
        task = BenchmarkTask(
            task_id="task_123",
            benchmark_name="infiagent",
            query="What is the average revenue?",
            ground_truth=45.2,
            eval_type="float_tol",
            metadata={"source": "unit_test", "difficulty": "easy"}
        )
        assert task.task_id == "task_123"
        assert task.benchmark_name == "infiagent"
        assert task.query == "What is the average revenue?"
        assert task.ground_truth == 45.2
        assert task.eval_type == "float_tol"
        assert task.metadata["difficulty"] == "easy"

    def test_benchmark_task_svamp_valid(self):
        """Task accepts svamp as valid benchmark_name."""
        task = BenchmarkTask(
            task_id="svamp_001",
            benchmark_name="svamp",
            query="Solve 5 + 3.",
            ground_truth=8.0,
            eval_type="float_tol",
            metadata={"type": "Common-Addition"}
        )
        assert task.benchmark_name == "svamp"
        assert task.eval_type == "float_tol"

    def test_benchmark_task_validation_missing_required(self):
        """Missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            # Missing task_id, query, ground_truth, eval_type
            BenchmarkTask(benchmark_name="bird_sql")  # type: ignore

    def test_benchmark_task_invalid_benchmark_name(self):
        """Unsupported benchmark name raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkTask(
                task_id="t1",
                benchmark_name="unsupported_benchmark",  # type: ignore
                query="Sample query",
                ground_truth=10,
                eval_type="exact"
            )

    def test_benchmark_task_invalid_eval_type(self):
        """Unsupported eval_type raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkTask(
                task_id="t1",
                benchmark_name="databench",
                query="Sample query",
                ground_truth=10,
                eval_type="invalid_eval_strategy"  # type: ignore
            )

    def test_benchmark_task_serialization_roundtrip(self):
        """to_dict() and from_dict() correctly serialize and reconstruct."""
        original = BenchmarkTask(
            task_id="t_roundtrip",
            benchmark_name="synthetic",
            query="Select * from sales;",
            db_path="/tmp/sales.db",
            table_path=None,
            ground_truth=[(1, "Product A", 100.0)],
            eval_type="sql_multiset",
            metadata={"seed": 42}
        )
        d = original.to_dict()
        assert isinstance(d, dict)
        assert d["task_id"] == "t_roundtrip"
        assert d["eval_type"] == "sql_multiset"

        reconstructed = BenchmarkTask.from_dict(d)
        assert reconstructed.task_id == original.task_id
        assert reconstructed.benchmark_name == original.benchmark_name
        assert reconstructed.ground_truth == original.ground_truth
        assert reconstructed.metadata == original.metadata


class TestTaskSplitEnum:
    """Test TaskSplit enum values."""

    def test_task_split_values(self):
        assert TaskSplit.TRAIN.value == "train"
        assert TaskSplit.VALIDATION.value == "validation"
        assert TaskSplit.TEST.value == "test"
        assert TaskSplit.DEV.value == "dev"
        assert TaskSplit.LITE.value == "lite"
        assert TaskSplit.FULL.value == "full"


class DummyLoader(BaseDatasetLoader):
    """Concrete implementation of BaseDatasetLoader for testing ABC contract."""

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        tasks = [
            BenchmarkTask(
                task_id="dummy_01",
                benchmark_name="synthetic",
                query="Dummy query",
                ground_truth=1,
                eval_type="exact"
            )
        ]
        if limit is not None:
            return tasks[:limit]
        return tasks

    def get_task(self, task_id: str) -> BenchmarkTask:
        if task_id == "dummy_01":
            return self.load_tasks()[0]
        raise KeyError(f"Task {task_id} not found")

    def get_manifest(self) -> Dict[str, Any]:
        return {"total_tasks": 1, "split": self.split.value}


class TestBaseDatasetLoaderABC:
    """Test BaseDatasetLoader interface and ABC enforcement."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            BaseDatasetLoader()  # type: ignore

    def test_dummy_loader_methods(self):
        loader = DummyLoader(split=TaskSplit.TEST)
        assert loader.split == TaskSplit.TEST
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "dummy_01"

        task = loader.get_task("dummy_01")
        assert task.task_id == "dummy_01"

        manifest = loader.get_manifest()
        assert manifest["total_tasks"] == 1
        assert manifest["split"] == "test"

        with pytest.raises(KeyError):
            loader.get_task("non_existent")
