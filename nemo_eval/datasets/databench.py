"""
nemo_eval.datasets.databench
============================
DataBench tabular question-answering dataset loader with 4 semantic type
categorization (Scalar, Boolean, List/Set, Table).
"""

from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


class DataBenchSemanticType(str, Enum):
    """The 4 canonical semantic question types in DataBench."""
    SCALAR = "Scalar"
    BOOLEAN = "Boolean"
    LIST_SET = "List/Set"
    TABLE = "Table"


def categorize_semantic_type(
    question: str,
    ground_truth: Any = None,
    explicit_type: Optional[str] = None
) -> DataBenchSemanticType:
    """
    Classify a DataBench question into one of 4 semantic types.
    """
    if explicit_type:
        type_str = explicit_type.strip().lower()
        if type_str in ("scalar", "number", "float", "int", "string", "category", "date"):
            return DataBenchSemanticType.SCALAR
        elif type_str in ("boolean", "bool", "binary", "yes/no"):
            return DataBenchSemanticType.BOOLEAN
        elif type_str in ("list/set", "list", "set", "array", "collection", "multiset"):
            return DataBenchSemanticType.LIST_SET
        elif type_str in ("table", "dataframe", "subtable", "matrix"):
            return DataBenchSemanticType.TABLE

    # Check ground truth type
    if isinstance(ground_truth, bool):
        return DataBenchSemanticType.BOOLEAN
    elif isinstance(ground_truth, (int, float)):
        return DataBenchSemanticType.SCALAR
    elif isinstance(ground_truth, (list, tuple, set)):
        if all(isinstance(x, (dict, list, tuple)) for x in ground_truth):
            return DataBenchSemanticType.TABLE
        return DataBenchSemanticType.LIST_SET
    elif isinstance(ground_truth, dict):
        # Likely a dataframe records or columnar dict
        return DataBenchSemanticType.TABLE

    # String / Text heuristics on question and ground_truth
    gt_str = str(ground_truth).strip().lower() if ground_truth is not None else ""
    if gt_str in ("true", "false", "yes", "no"):
        return DataBenchSemanticType.BOOLEAN

    q_lower = question.lower()
    if q_lower.startswith(("is there", "are there", "does ", "do ", "has ", "have ", "was ", "were ", "is it ", "can ")):
        return DataBenchSemanticType.BOOLEAN
    if q_lower.startswith(("list ", "name all ", "which ", "what are the names", "give all")):
        return DataBenchSemanticType.LIST_SET
    if "table" in q_lower or "cross-tab" in q_lower or "breakdown" in q_lower or "pivot" in q_lower:
        return DataBenchSemanticType.TABLE

    return DataBenchSemanticType.SCALAR


def map_semantic_type_to_eval_strategy(
    semantic_type: Union[str, DataBenchSemanticType],
    ground_truth: Any
) -> str:
    """
    Map semantic type to polymorphic evaluation strategy:
    - Scalar (numeric) -> float_tol
    - Scalar (text/date) -> exact
    - Boolean -> exact
    - List/Set -> exact
    - Table -> dataframe_diff
    """
    st = semantic_type.value if isinstance(semantic_type, DataBenchSemanticType) else semantic_type
    
    if st == DataBenchSemanticType.BOOLEAN.value or st == "Boolean":
        return "exact"
    elif st == DataBenchSemanticType.TABLE.value or st == "Table":
        return "dataframe_diff"
    elif st == DataBenchSemanticType.LIST_SET.value or st == "List/Set":
        return "exact"
    else: # Scalar
        if isinstance(ground_truth, (int, float)):
            return "float_tol"
        elif isinstance(ground_truth, str):
            # Check if parseable as numeric
            cleaned = ground_truth.strip().replace("$", "").replace("%", "").replace(",", "")
            try:
                float(cleaned)
                return "float_tol"
            except ValueError:
                return "exact"
        return "exact"


class DataBenchLoader(BaseDatasetLoader):
    """
    Dataset loader for DataBench tabular reasoning tasks.
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: TaskSplit = TaskSplit.LITE,
        tasks_data: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(dataset_root=dataset_root, split=split)
        self._tasks_data = tasks_data
        self._cache: Optional[List[BenchmarkTask]] = None

    def _parse_task_item(self, item: Dict[str, Any], idx: int = 0) -> BenchmarkTask:
        """Parse raw task dictionary into canonical BenchmarkTask."""
        task_id = str(item.get("task_id", item.get("id", f"db_{idx}")))
        query = item.get("question", item.get("query", ""))
        ground_truth = item.get("answer", item.get("ground_truth"))
        
        sem_type = categorize_semantic_type(
            question=query,
            ground_truth=ground_truth,
            explicit_type=item.get("semantic_type", item.get("type"))
        )
        
        eval_type = item.get("eval_type") or map_semantic_type_to_eval_strategy(
            sem_type, ground_truth
        )

        table_path = item.get("table_path", item.get("dataset_path", item.get("file_path")))
        if table_path and self.dataset_root and not os.path.isabs(table_path):
            table_path = os.path.join(self.dataset_root, table_path)

        metadata = {
            "semantic_type": sem_type.value,
            "dataset_name": item.get("dataset_name", "databench"),
            "split": str(self.split.value),
        }
        if "metadata" in item and isinstance(item["metadata"], dict):
            metadata.update(item["metadata"])

        return BenchmarkTask(
            task_id=task_id,
            benchmark_name="databench",
            query=query,
            context_schema=item.get("context_schema"),
            db_path=None,
            table_path=table_path,
            ground_truth=ground_truth,
            eval_type=eval_type,
            metadata=metadata
        )

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load and parse tasks for the active split with optional sample limit."""
        if self._cache is not None:
            tasks = self._cache
        else:
            tasks = []
            if self._tasks_data is not None:
                for i, item in enumerate(self._tasks_data):
                    tasks.append(self._parse_task_item(item, i))
            elif self.dataset_root and os.path.exists(self.dataset_root):
                candidate_files = [
                    os.path.join(self.dataset_root, f"{self.split.value}.json"),
                    os.path.join(self.dataset_root, f"{self.split.value}.jsonl"),
                    os.path.join(self.dataset_root, "questions.json"),
                ]
                chosen_file = None
                for cf in candidate_files:
                    if os.path.exists(cf):
                        chosen_file = cf
                        break

                if chosen_file:
                    if chosen_file.endswith(".jsonl"):
                        with open(chosen_file, "r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                line = line.strip()
                                if line:
                                    tasks.append(self._parse_task_item(json.loads(line), i))
                    else:
                        with open(chosen_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for i, item in enumerate(data):
                                    tasks.append(self._parse_task_item(item, i))
                            elif isinstance(data, dict) and "tasks" in data:
                                for i, item in enumerate(data["tasks"]):
                                    tasks.append(self._parse_task_item(item, i))

            self._cache = tasks

        if limit is not None and limit >= 0:
            return tasks[:limit]
        return list(tasks)

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        tasks = self.load_tasks()
        for t in tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task with ID '{task_id}' not found in DataBenchLoader.")

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        tasks = self.load_tasks()
        semantic_counts = {}
        for t in tasks:
            st = t.metadata.get("semantic_type", "unspecified")
            semantic_counts[st] = semantic_counts.get(st, 0) + 1

        return {
            "benchmark_name": "databench",
            "split": self.split.value,
            "total_tasks": len(tasks),
            "semantic_type_breakdown": semantic_counts,
            "dataset_root": self.dataset_root,
        }
