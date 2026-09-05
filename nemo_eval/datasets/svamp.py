"""
nemo_eval.datasets.svamp
========================
SVAMP (Simple Variations on Arithmetic Math Word Problems) benchmark dataset loader.

Ingests 1,000 challenge word problems with arithmetic variations and float/integer tolerances.
Evaluator type: float_tol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


class SVAMPLoader(BaseDatasetLoader):
    """
    Loader for the SVAMP benchmark dataset.

    Provides 1,000 challenge word problems evaluating sensitivity of reasoning models
    to surface variations in arithmetic questions.
    """

    TYPES: List[str] = [
        "Common-Addition",
        "Common-Subtraction",
        "Common-Multiplication",
        "Common-Division",
    ]

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: Union[TaskSplit, str] = TaskSplit.TEST,
        max_tasks: Optional[int] = None,
        use_fixture: bool = True,
        fixture_name: Optional[str] = None,
        category: Optional[str] = None,
    ):
        split_enum = TaskSplit(split) if isinstance(split, str) and split in TaskSplit._value2member_map_ else (
            split if isinstance(split, TaskSplit) else TaskSplit.TEST
        )
        super().__init__(dataset_root=dataset_root, split=split_enum)
        self.max_tasks = max_tasks
        self.use_fixture = use_fixture
        self.fixture_name = fixture_name or "svamp_1000.jsonl"
        self.category = category
        self._raw_tasks: Optional[List[BenchmarkTask]] = None

    def _get_fixture_path(self) -> Path:
        if self.dataset_root and os.path.exists(os.path.join(self.dataset_root, self.fixture_name)):
            return Path(self.dataset_root) / self.fixture_name
        fixtures_dir = Path(__file__).parent / "fixtures"
        return fixtures_dir / self.fixture_name

    def _fallback_load_hf(self) -> List[BenchmarkTask]:
        """Fallback to HuggingFace if offline fixture is absent."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise FileNotFoundError(
                f"SVAMP fixture file '{self._get_fixture_path()}' not found, and 'datasets' library is unavailable."
            )

        try:
            ds_train = load_dataset("ChilleD/SVAMP", split="train", trust_remote_code=False)
            ds_test = load_dataset("ChilleD/SVAMP", split="test", trust_remote_code=False)
            records = []
            idx = 1
            for row in list(ds_train) + list(ds_test):
                raw_id = row.get("ID", f"chal_{idx}")
                task_id = f"svamp_{raw_id}" if not str(raw_id).startswith("svamp_") else str(raw_id)
                body = str(row.get("Body", "")).strip()
                question = str(row.get("Question", "")).strip()
                query_text = f"{body} {question}".strip()
                answer = row.get("Answer", "")
                try:
                    ans_float = float(answer)
                    ans_str = str(int(ans_float)) if ans_float.is_integer() else str(ans_float)
                except (ValueError, TypeError):
                    ans_str = str(answer)

                t_type = row.get("Type", "")
                if t_type == "Common-Divison":
                    t_type = "Common-Division"

                records.append(
                    BenchmarkTask(
                        task_id=task_id,
                        benchmark_name="svamp",
                        query=f"{query_text}\n\nSolve this step-by-step using Python code. Put your final answer within \\boxed{{}}.",
                        ground_truth=f"\\boxed{{{ans_str}}}",
                        eval_type="float_tol",
                        db_path=None,
                        table_path=None,
                        metadata={
                            "source": "svamp",
                            "type": t_type,
                            "category": t_type,
                            "subdiscipline": t_type,
                            "body": body,
                            "question": question,
                            "equation": row.get("Equation", ""),
                            "split": "test",
                        },
                    )
                )
                idx += 1
            return records
        except Exception as e:
            raise FileNotFoundError(
                f"SVAMP fixture file '{self._get_fixture_path()}' not found and failed to load from HuggingFace: {e}"
            )

    def _load_raw_tasks(self) -> List[BenchmarkTask]:
        if self._raw_tasks is not None:
            return self._raw_tasks

        fixture_file = self._get_fixture_path()
        tasks: List[BenchmarkTask] = []

        if self.use_fixture and fixture_file.exists():
            with open(fixture_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        tasks.append(BenchmarkTask.from_dict(record))
        else:
            tasks = self._fallback_load_hf()

        self._raw_tasks = tasks
        return self._raw_tasks

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load and parse SVAMP tasks with optional category filtering and limit."""
        all_tasks = self._load_raw_tasks()
        filtered = all_tasks

        if self.category:
            cat_clean = self.category.lower().strip().replace(" ", "-").replace("_", "-")
            filtered = [
                t for t in filtered
                if t.subdiscipline.lower().strip().replace(" ", "-").replace("_", "-") == cat_clean
                or str(t.metadata.get("type", "")).lower().strip().replace(" ", "-").replace("_", "-") == cat_clean
                or str(t.metadata.get("category", "")).lower().strip().replace(" ", "-").replace("_", "-") == cat_clean
            ]

        effective_limit = limit if limit is not None else self.max_tasks
        if effective_limit is not None:
            if effective_limit <= 0:
                return []
            filtered = filtered[:effective_limit]

        return filtered

    def load(
        self,
        split: Optional[Union[TaskSplit, str]] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[BenchmarkTask]:
        """Convenience alias conforming to PROJECT.md interface contract."""
        if split is not None:
            if isinstance(split, str) and split not in ("test", "train", "val", "validation", "dev", "lite", "full"):
                raise ValueError(f"Unknown split '{split}'")
            if isinstance(split, str) and split in TaskSplit._value2member_map_:
                self.split = TaskSplit(split)
        if category is not None:
            self.category = category
        return self.load_tasks(limit=limit)

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        all_tasks = self._load_raw_tasks()
        for t in all_tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task '{task_id}' not found in SVAMP dataset.")

    def get_types(self) -> List[str]:
        """Return the list of standard SVAMP variation types."""
        return list(self.TYPES)

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        all_tasks = self._load_raw_tasks()
        return {
            "benchmark_name": "svamp",
            "total_tasks": len(all_tasks),
            "types": self.TYPES,
            "split": str(self.split.value if isinstance(self.split, TaskSplit) else self.split),
            "source": "svamp",
            "offline_fixture": str(self._get_fixture_path()),
        }
