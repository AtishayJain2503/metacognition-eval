"""
nemo_eval.datasets.base
=======================
Canonical task representations and abstract loader contracts for benchmark suites.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class TaskSplit(str, Enum):
    """Supported dataset splits and evaluation profiles."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    DEV = "dev"
    LITE = "lite"
    FULL = "full"


class BenchmarkTask(BaseModel):
    """Canonical task representation contract for all benchmark suites."""
    model_config = ConfigDict(extra="ignore")

    task_id: str = Field(..., description="Unique task identifier across benchmarks.")
    benchmark_name: Literal["infiagent", "bird_sql", "databench", "synthetic"] = Field(
        ..., description="Name of the benchmark suite."
    )
    query: str = Field(..., description="The natural language instruction / query provided to the agent.")
    context_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Schema metadata (DDL, table columns, data types, sample rows)."
    )
    db_path: Optional[str] = Field(
        default=None, description="Absolute or relative path to target SQLite database."
    )
    table_path: Optional[str] = Field(
        default=None, description="Absolute or relative path to tabular data file (CSV, Parquet, TSV)."
    )
    ground_truth: Any = Field(
        ..., description="Ground truth answer (scalar, boolean, list, SQL string, DataFrame dict/records)."
    )
    eval_type: Literal["exact", "float_tol", "sql_multiset", "dataframe_diff"] = Field(
        ..., description="Evaluation strategy used to compare predicted response against ground truth."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific metadata (difficulty, semantic_type, evidence, golden_sql, etc.)."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to a JSON-compatible dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkTask":
        """Instantiate a BenchmarkTask from a dictionary."""
        return cls.model_validate(data)


class BaseDatasetLoader(ABC):
    """Abstract base class for benchmark dataset loaders."""

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: TaskSplit = TaskSplit.TEST
    ):
        self.dataset_root = dataset_root
        self.split = split

    @abstractmethod
    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load and parse tasks for the active split with optional sample limit."""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        pass

    @abstractmethod
    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary including total tasks, splits, and schema inventory."""
        pass
