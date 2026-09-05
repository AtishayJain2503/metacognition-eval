"""
nemo_eval.datasets.math
=======================
Hendrycks MATH benchmark dataset loader (50 representative samples across 7 subjects).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


class MATHLoader(BaseDatasetLoader):
    """
    Loader for the Hendrycks MATH benchmark dataset.
    
    Provides 50 representative competition & school mathematics problems across 7 core subjects:
    Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus.
    """

    SUBJECTS: List[str] = [
        "Algebra",
        "Counting & Probability",
        "Geometry",
        "Intermediate Algebra",
        "Number Theory",
        "Prealgebra",
        "Precalculus",
    ]

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: Union[TaskSplit, str] = TaskSplit.TEST,
        subject: Optional[str] = None,
        max_tasks: Optional[int] = 50,
        use_fixture: bool = True,
        use_1000: bool = False,
        fixture_name: Optional[str] = None,
    ):
        split_enum = TaskSplit(split) if isinstance(split, str) and split in TaskSplit._value2member_map_ else (
            split if isinstance(split, TaskSplit) else TaskSplit.TEST
        )
        super().__init__(dataset_root=dataset_root, split=split_enum)
        if use_1000 and max_tasks == 50:
            max_tasks = None
        self.subject = subject
        self.max_tasks = max_tasks
        self.use_fixture = use_fixture
        self.use_1000 = use_1000
        self.fixture_name = fixture_name
        self._raw_tasks: Optional[List[BenchmarkTask]] = None

    def _resolve_fixture_filename(self) -> str:
        if self.fixture_name:
            return self.fixture_name
        if self.use_1000 or (self.max_tasks is not None and self.max_tasks > 50):
            fixtures_dir = Path(self.dataset_root) if self.dataset_root else Path(__file__).parent / "fixtures"
            if (fixtures_dir / "math_1000.jsonl").exists():
                return "math_1000.jsonl"
        return "math_tasks.jsonl"

    def _get_fixture_path(self) -> Path:
        filename = self._resolve_fixture_filename()
        if self.dataset_root and os.path.exists(os.path.join(self.dataset_root, filename)):
            return Path(self.dataset_root) / filename
        fixtures_dir = Path(__file__).parent / "fixtures"
        return fixtures_dir / filename

    def _load_raw_tasks(self) -> List[BenchmarkTask]:
        if self._raw_tasks is not None:
            return self._raw_tasks

        fixture_file = self._get_fixture_path()
        tasks: List[BenchmarkTask] = []

        if fixture_file.exists():
            with open(fixture_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        tasks.append(BenchmarkTask.from_dict(record))
        else:
            # Fallback inline generation if fixture file is missing
            from nemo_eval.datasets.fixtures.generate_fixtures import generate_math_fixtures
            records = generate_math_fixtures()
            for r in records:
                tasks.append(BenchmarkTask.from_dict(r))

        self._raw_tasks = tasks
        return self._raw_tasks

    def load_tasks(self, limit: Optional[int] = None, use_1000: Optional[bool] = None) -> List[BenchmarkTask]:
        """Load and parse MATH tasks with optional filtering and sample limits."""
        if use_1000 is not None and use_1000 != self.use_1000:
            self.use_1000 = use_1000
            if use_1000 and self.max_tasks == 50:
                self.max_tasks = None
            self._raw_tasks = None
        all_tasks = self._load_raw_tasks()
        filtered = all_tasks

        if self.subject:
            subj_clean = self.subject.lower().strip()
            filtered = [
                t for t in filtered
                if t.subdiscipline.lower().strip() == subj_clean
                or str(t.metadata.get("subject", "")).lower().strip() == subj_clean
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
        subject: Optional[str] = None,
        use_1000: Optional[bool] = None,
    ) -> List[BenchmarkTask]:
        """Convenience alias conforming to PROJECT.md interface contract."""
        if split is not None:
            if isinstance(split, str) and split not in ("test", "train", "val", "validation", "dev", "lite", "full"):
                raise ValueError(f"Unknown split '{split}'")
            if isinstance(split, str) and split in TaskSplit._value2member_map_:
                self.split = TaskSplit(split)
        if subject is not None:
            self.subject = subject
        return self.load_tasks(limit=limit, use_1000=use_1000)

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        all_tasks = self._load_raw_tasks()
        for t in all_tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task '{task_id}' not found in MATH dataset.")

    def get_subjects(self) -> List[str]:
        """Return the list of standard Hendrycks MATH subjects."""
        return list(self.SUBJECTS)

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        all_tasks = self._load_raw_tasks()
        return {
            "benchmark_name": "math",
            "total_tasks": len(all_tasks),
            "subjects": self.SUBJECTS,
            "split": str(self.split.value if isinstance(self.split, TaskSplit) else self.split),
            "source": "hendrycks_math",
            "offline_fixture": str(self._get_fixture_path()),
        }
