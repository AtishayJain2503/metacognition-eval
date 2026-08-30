"""
nemo_eval.datasets.base
=======================
Canonical task representations and abstract loader contracts for benchmark suites.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "dataset_name" in data and "benchmark_name" not in data:
                data["benchmark_name"] = data["dataset_name"]
            if "problem_text" in data and "query" not in data:
                data["query"] = data["problem_text"]
            if "subdiscipline" in data and "metadata" not in data:
                data["metadata"] = {"subdiscipline": data["subdiscipline"]}
            elif "subdiscipline" in data and isinstance(data.get("metadata"), dict):
                data["metadata"].setdefault("subdiscipline", data["subdiscipline"])
        return data

    task_id: str = Field(..., description="Unique task identifier across benchmarks.")
    benchmark_name: Literal[
        "infiagent", "bird_sql", "databench", "synthetic", "gsm8k", "math", "putnam", "lila"
    ] = Field(
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
        ..., description="Ground truth answer (scalar, boolean, list, SQL string, DataFrame dict/records, LaTeX/math expression)."
    )
    eval_type: Literal[
        "exact", "float_tol", "sql_multiset", "dataframe_diff", "math_symbolic", "fraction", "set"
    ] = Field(
        ..., description="Evaluation strategy used to compare predicted response against ground truth."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific metadata (difficulty, semantic_type, evidence, golden_sql, etc.)."
    )

    @property
    def dataset_name(self) -> str:
        """Alias for benchmark_name conforming to PROJECT.md interface contract."""
        return self.benchmark_name

    @property
    def problem_text(self) -> str:
        """Alias for query conforming to PROJECT.md interface contract."""
        return self.query

    @property
    def subdiscipline(self) -> str:
        """Extract subdiscipline, subject, or category from metadata."""
        return str(
            self.metadata.get(
                "subdiscipline",
                self.metadata.get("subject", self.metadata.get("category", self.metadata.get("subcategory", "")))
            )
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

    def load(self, split: Optional[str] = None, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Convenience alias for load_tasks conforming to PROJECT.md interface contract."""
        if split is not None:
            self.split = TaskSplit(split) if isinstance(split, str) else split
        return self.load_tasks(limit=limit)

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
