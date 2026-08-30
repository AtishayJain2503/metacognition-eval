"""
nemo_eval.datasets.lila
=======================
AllenAI Lila benchmark dataset loader (7 core mathematical reasoning subcategories x 50 tasks = 350 total).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


class LilaLoader(BaseDatasetLoader):
    """
    Loader for the AllenAI Lila multi-domain mathematical and scientific reasoning benchmark.
    
    Provides 350 curated tasks spanning all 7 core mathematical reasoning categories:
    Arithmetic, Algebra, Calculus, Geometry, Combinatorics, Physics, and Statistics (50 tasks each).
    """

    SUBCATEGORIES: List[str] = [
        "arithmetic",
        "algebra",
        "calculus",
        "geometry",
        "combinatorics",
        "physics",
        "statistics",
    ]

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: Union[TaskSplit, str] = TaskSplit.TEST,
        subcategories: Optional[List[str]] = None,
        max_tasks_per_category: Optional[int] = 50,
        use_fixture: bool = True,
    ):
        split_enum = TaskSplit(split) if isinstance(split, str) and split in TaskSplit._value2member_map_ else (
            split if isinstance(split, TaskSplit) else TaskSplit.TEST
        )
        super().__init__(dataset_root=dataset_root, split=split_enum)
        self.subcategories = [s.lower().strip() for s in subcategories] if subcategories else list(self.SUBCATEGORIES)
        self.max_tasks_per_category = max_tasks_per_category
        self.use_fixture = use_fixture
        self._raw_tasks: Optional[List[BenchmarkTask]] = None

    def _get_fixture_path(self) -> Path:
        if self.dataset_root and os.path.exists(os.path.join(self.dataset_root, "lila_tasks.jsonl")):
            return Path(self.dataset_root) / "lila_tasks.jsonl"
        fixtures_dir = Path(__file__).parent / "fixtures"
        return fixtures_dir / "lila_tasks.jsonl"

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
            from nemo_eval.datasets.fixtures.generate_fixtures import generate_lila_fixtures
            records = generate_lila_fixtures()
            for r in records:
                tasks.append(BenchmarkTask.from_dict(r))

        self._raw_tasks = tasks
        return self._raw_tasks

    def load_tasks(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Load and parse Lila tasks filtered by subcategories and sample limits."""
        all_tasks = self._load_raw_tasks()
        allowed_cats = {s.lower().strip() for s in self.subcategories}
        
        # Group tasks by category to respect max_tasks_per_category if set
        filtered: List[BenchmarkTask] = []
        counts_by_cat: Dict[str, int] = {c: 0 for c in self.SUBCATEGORIES}

        for t in all_tasks:
            cat = str(t.metadata.get("subcategory", t.subdiscipline)).lower().strip()
            if cat in allowed_cats:
                if self.max_tasks_per_category is None or counts_by_cat[cat] < self.max_tasks_per_category:
                    filtered.append(t)
                    counts_by_cat[cat] += 1

        if limit is not None:
            if limit <= 0:
                return []
            filtered = filtered[:limit]

        return filtered

    def load(
        self,
        split: Optional[Union[TaskSplit, str]] = None,
        limit: Optional[int] = None,
        subcategories: Optional[List[str]] = None,
        subdiscipline: Optional[str] = None,
    ) -> List[BenchmarkTask]:
        """Convenience alias conforming to PROJECT.md interface contract."""
        if split is not None:
            if isinstance(split, str) and split not in ("test", "train", "val", "validation", "dev", "lite", "full"):
                raise ValueError(f"Unknown split '{split}'")
            if isinstance(split, str) and split in TaskSplit._value2member_map_:
                self.split = TaskSplit(split)
        if subcategories is not None:
            self.subcategories = [s.lower().strip() for s in subcategories]
        if subdiscipline is not None:
            self.subcategories = [subdiscipline.lower().strip()]
        return self.load_tasks(limit=limit)

    def load_category(self, category: str, limit: Optional[int] = 50) -> List[BenchmarkTask]:
        """Load tasks for a specific Lila subcategory."""
        cat_clean = category.lower().strip()
        all_tasks = self._load_raw_tasks()
        filtered = [
            t for t in all_tasks
            if str(t.metadata.get("subcategory", t.subdiscipline)).lower().strip() == cat_clean
        ]
        if limit is not None:
            if limit <= 0:
                return []
            filtered = filtered[:limit]
        return filtered

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Retrieve a single task by unique identifier."""
        all_tasks = self._load_raw_tasks()
        for t in all_tasks:
            if t.task_id == task_id:
                return t
        raise KeyError(f"Task '{task_id}' not found in Lila dataset.")

    def get_subcategories(self) -> List[str]:
        """Return list of standard Lila subcategories."""
        return list(self.SUBCATEGORIES)

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        all_tasks = self._load_raw_tasks()
        return {
            "benchmark_name": "lila",
            "total_tasks": len(all_tasks),
            "subcategories": self.SUBCATEGORIES,
            "split": str(self.split.value if isinstance(self.split, TaskSplit) else self.split),
            "source": "allenai_lila",
            "offline_fixture": str(self._get_fixture_path()),
        }
