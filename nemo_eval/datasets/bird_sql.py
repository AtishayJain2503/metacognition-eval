"""
nemo_eval.datasets.bird_sql
===========================
BIRD-SQL and Spider 2.0-Lite text-to-SQL dataset loader with schema metadata
binding and domain evidence dictionary injection.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BaseDatasetLoader, BenchmarkTask, TaskSplit


def format_bird_sql_prompt(
    query: str,
    evidence: Optional[str] = None,
    schema_ddl: Optional[str] = None
) -> str:
    """
    Format BIRD-SQL / Spider 2.0 query prompt with injected schema structure
    and domain evidence dictionary.
    """
    sections = []
    if schema_ddl and schema_ddl.strip():
        sections.append(f"Database Schema:\n{schema_ddl.strip()}")
    if evidence and evidence.strip():
        sections.append(f"External Domain Evidence:\n{evidence.strip()}")
    sections.append(f"Question: {query.strip()}")
    return "\n\n".join(sections)


def normalize_sql_query(sql: str) -> str:
    """Normalize SQL query string by stripping whitespace and trailing semicolon."""
    if not sql:
        return ""
    cleaned = sql.strip()
    cleaned = re.sub(r";\s*$", "", cleaned)
    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class BirdSqlLoader(BaseDatasetLoader):
    """
    Dataset loader for BIRD-SQL and Spider 2.0-Lite benchmark suites.
    """

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        split: TaskSplit = TaskSplit.DEV,
        databases_dir: Optional[str] = None,
        tasks_data: Optional[List[Dict[str, Any]]] = None,
        tables_data: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(dataset_root=dataset_root, split=split)
        self.databases_dir = databases_dir or (
            os.path.join(dataset_root, "databases") if dataset_root else None
        )
        self._tasks_data = tasks_data
        self._tables_data = tables_data
        self._schemas_by_db_id: Dict[str, Any] = {}
        self._cache: Optional[List[BenchmarkTask]] = None
        self._load_tables_schema()

    def _load_tables_schema(self) -> None:
        """Load schemas from tables.json if available."""
        if self._tables_data:
            for item in self._tables_data:
                db_id = item.get("db_id")
                if db_id:
                    self._schemas_by_db_id[db_id] = item
        elif self.dataset_root and os.path.exists(self.dataset_root):
            tables_path = os.path.join(self.dataset_root, "tables.json")
            if os.path.exists(tables_path):
                try:
                    with open(tables_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                db_id = item.get("db_id")
                                if db_id:
                                    self._schemas_by_db_id[db_id] = item
                except Exception:
                    pass

    def _resolve_db_path(self, db_id: Optional[str]) -> Optional[str]:
        """Resolve path to SQLite database file."""
        if not db_id:
            return None
        
        # Check databases_dir / db_id / db_id.sqlite
        if self.databases_dir and os.path.exists(self.databases_dir):
            candidates = [
                os.path.join(self.databases_dir, db_id, f"{db_id}.sqlite"),
                os.path.join(self.databases_dir, f"{db_id}.sqlite"),
                os.path.join(self.databases_dir, db_id, f"{db_id}.db"),
                os.path.join(self.databases_dir, f"{db_id}.db"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c

        if self.dataset_root and os.path.exists(self.dataset_root):
            candidates = [
                os.path.join(self.dataset_root, db_id, f"{db_id}.sqlite"),
                os.path.join(self.dataset_root, f"{db_id}.sqlite"),
                os.path.join(self.dataset_root, "databases", db_id, f"{db_id}.sqlite"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c

        return f"{db_id}.sqlite"

    def _parse_task_item(self, item: Dict[str, Any], idx: int = 0) -> BenchmarkTask:
        """Parse raw task dictionary into canonical BenchmarkTask."""
        task_id = str(item.get("question_id", item.get("task_id", item.get("id", f"bird_{idx}"))))
        db_id = item.get("db_id", "")
        query = item.get("question", item.get("query", ""))
        evidence = item.get("evidence", "")
        golden_sql = normalize_sql_query(item.get("SQL", item.get("golden_sql", item.get("query_sql", ""))))
        difficulty = item.get("difficulty", "moderate")
        
        db_path = item.get("db_path") or self._resolve_db_path(db_id)
        schema_info = self._schemas_by_db_id.get(db_id, item.get("context_schema"))

        # Build metadata
        metadata = {
            "db_id": db_id,
            "evidence": evidence,
            "golden_sql": golden_sql,
            "difficulty": difficulty,
            "split": str(self.split.value),
        }
        if "metadata" in item and isinstance(item["metadata"], dict):
            metadata.update(item["metadata"])

        # Ground truth can be the golden SQL query string or result set representation
        ground_truth = item.get("ground_truth", golden_sql)

        return BenchmarkTask(
            task_id=task_id,
            benchmark_name="bird_sql",
            query=query,
            context_schema=schema_info if isinstance(schema_info, dict) else None,
            db_path=db_path,
            table_path=None,
            ground_truth=ground_truth,
            eval_type="sql_multiset",
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
                    os.path.join(self.dataset_root, f"{self.split.value}_databases.json"),
                    os.path.join(self.dataset_root, f"dev.json" if self.split == TaskSplit.DEV else f"train.json"),
                ]
                chosen_file = None
                for cf in candidate_files:
                    if os.path.exists(cf):
                        chosen_file = cf
                        break

                if chosen_file:
                    with open(chosen_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for i, item in enumerate(data):
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
        raise KeyError(f"Task with ID '{task_id}' not found in BirdSqlLoader.")

    def get_manifest(self) -> Dict[str, Any]:
        """Return dataset metadata summary."""
        tasks = self.load_tasks()
        difficulties = {}
        for t in tasks:
            diff = t.metadata.get("difficulty", "unspecified")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "benchmark_name": "bird_sql",
            "split": self.split.value,
            "total_tasks": len(tasks),
            "difficulty_breakdown": difficulties,
            "database_count": len(set(t.metadata.get("db_id") for t in tasks if t.metadata.get("db_id"))),
            "dataset_root": self.dataset_root,
        }
