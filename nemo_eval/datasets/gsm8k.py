"""
nemo_eval.datasets.gsm8k
========================
GSM8K (Grade School Math 8K) dataset loader.

Loads GSM8K benchmark problems with exact integer ground-truth extraction.
Default mode loads from offline fixture (fixtures/gsm8k_1000.jsonl),
with graceful fallback to HuggingFace datasets library.
Evaluation type: float_tol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


def _extract_answer(solution_text: str) -> Optional[int]:
    """Extract the integer answer after '####' in GSM8K solution strings."""
    match = re.search(r"####\s*([\-0-9,]+)", solution_text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return int(raw)
        except ValueError:
            return None
    return None


class GSM8KLoader(BaseDatasetLoader):
    """
    Loader for the GSM8K benchmark dataset.

    Inherits from BaseDatasetLoader to conform to benchmark runner contracts.
    Loads from hermetic offline JSONL fixtures by default with HuggingFace fallback.
    """

    HF_DATASET_ID = "openai/gsm8k"
    HF_CONFIG = "main"

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: Union[TaskSplit, str] = TaskSplit.TEST,
        max_tasks: Optional[int] = None,
        use_fixture: bool = True,
        fixture_name: Optional[str] = None,
    ):
        # Support positional split argument for backwards compatibility
        if isinstance(dataset_root, str) and dataset_root.lower() in (
            "test", "train", "val", "validation", "dev", "lite", "full"
        ):
            split = dataset_root
            dataset_root = None

        split_enum = TaskSplit(split) if isinstance(split, str) and split in TaskSplit._value2member_map_ else (
            split if isinstance(split, TaskSplit) else TaskSplit.TEST
        )
        super().__init__(dataset_root=dataset_root, split=split_enum)
        self.max_tasks = max_tasks
        self.use_fixture = use_fixture
        self.fixture_name = fixture_name or "gsm8k_1000.jsonl"
        self._raw_tasks: Optional[List[BenchmarkTask]] = None

    def _get_fixture_path(self) -> Path:
        if self.dataset_root and os.path.exists(os.path.join(self.dataset_root, self.fixture_name)):
            return Path(self.dataset_root) / self.fixture_name
        fixtures_dir = Path(__file__).parent / "fixtures"
        return fixtures_dir / self.fixture_name

    def _fallback_load_hf(self) -> List[BenchmarkTask]:
        """Download and parse GSM8K samples from HuggingFace when fixture is absent."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise FileNotFoundError(
                f"GSM8K fixture file '{self._get_fixture_path()}' not found, and 'datasets' library is not available."
            )

        active_split = self.split.value if isinstance(self.split, TaskSplit) else str(self.split)
        try:
            ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=active_split, trust_remote_code=False)
        except Exception as e:
            raise FileNotFoundError(
                f"GSM8K fixture file '{self._get_fixture_path()}' not found and failed to load from HuggingFace: {e}"
            )

        tasks: List[BenchmarkTask] = []
        for i, row in enumerate(ds):
            question: str = str(row["question"]).strip()
            answer_text: str = str(row["answer"]).strip()
            gt_int = _extract_answer(answer_text)

            if gt_int is None:
                continue

            task_id = f"gsm8k_{active_split}_{i:04d}"
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    benchmark_name="gsm8k",
                    query=(
                        f"{question}\n\n"
                        "Solve this step-by-step using Python code. "
                        "Your final answer must be an integer. Put your final answer within \\boxed{}."
                    ),
                    ground_truth=f"\\boxed{{{gt_int}}}",
                    eval_type="float_tol",
                    db_path=None,
                    table_path=None,
                    metadata={
                        "source": "gsm8k",
                        "original_answer": answer_text,
                        "split": active_split,
                        "index": i,
                        "tolerance": 0.5,
                    },
                )
            )

        return tasks

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
        """Load and parse GSM8K tasks with optional sample limits."""
        all_tasks = self._load_raw_tasks()
        effective_limit = limit if limit is not None else self.max_tasks
        if effective_limit is not None:
            if effective_limit <= 0:
                return []
            return all_tasks[:effective_limit]
        return all_tasks

    def load(
        self,
        split: Optional[Union[TaskSplit, str]] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkTask]:
        """Convenience alias conforming to PROJECT.md interface contract."""
        if split is not None:
            if isinstance(split, str) and split not in ("test", "train", "val", "validation", "dev", "lite", "full"):
                raise ValueError(f"Unknown split '{split}'")
            if isinstance(split, str) and split in TaskSplit._value2member_map_:
                if self.split != TaskSplit(split):
                    self.split = TaskSplit(split)
                    self._raw_tasks = None
        return self.load_tasks(limit=limit)

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        all_tasks = self._load_raw_tasks()
        for t in all_tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task '{task_id}' not found in GSM8K dataset.")

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        all_tasks = self._load_raw_tasks()
        return {
            "benchmark_name": "gsm8k",
            "total_tasks": len(all_tasks),
            "split": str(self.split.value if isinstance(self.split, TaskSplit) else self.split),
            "source": "gsm8k",
            "offline_fixture": str(self._get_fixture_path()),
        }
