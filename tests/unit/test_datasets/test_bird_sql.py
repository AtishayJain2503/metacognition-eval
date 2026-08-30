"""
tests.unit.test_datasets.test_bird_sql
======================================
Unit tests for BIRD-SQL / Spider 2.0 loader, prompt formatter, and schema bindings.
"""

import os
import pytest

from nemo_eval.datasets.base import TaskSplit
from nemo_eval.datasets.bird_sql import (
    BirdSqlLoader,
    format_bird_sql_prompt,
    normalize_sql_query,
)


class TestBirdSqlPromptAndNormalization:
    """Test prompt formatting and SQL query normalization."""

    def test_format_bird_sql_prompt_complete(self):
        prompt = format_bird_sql_prompt(
            query="Find the price of item X.",
            evidence="item X has product_id = 101",
            schema_ddl="CREATE TABLE products (product_id INT, price REAL);"
        )
        assert "Database Schema:" in prompt
        assert "CREATE TABLE products" in prompt
        assert "External Domain Evidence:" in prompt
        assert "item X has product_id = 101" in prompt
        assert "Question: Find the price of item X." in prompt

    def test_format_bird_sql_prompt_no_evidence(self):
        prompt = format_bird_sql_prompt(
            query="Count all customers.",
            evidence=None,
            schema_ddl="CREATE TABLE customers (id INT);"
        )
        assert "Database Schema:" in prompt
        assert "External Domain Evidence:" not in prompt
        assert "Question: Count all customers." in prompt

    def test_normalize_sql_query(self):
        raw = "  SELECT name,   price FROM products WHERE id = 1;  \n"
        norm = normalize_sql_query(raw)
        assert norm == "SELECT name, price FROM products WHERE id = 1"
        assert normalize_sql_query("") == ""


class TestBirdSqlLoader:
    """Test BirdSqlLoader loading, schema binding, and task metadata."""

    def test_load_from_directory(self, mock_bird_sql_dir):
        loader = BirdSqlLoader(dataset_root=mock_bird_sql_dir, split=TaskSplit.DEV)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        
        task = tasks[0]
        assert task.task_id == "bird_mock_01"
        assert task.benchmark_name == "bird_sql"
        assert task.eval_type == "sql_multiset"
        assert task.metadata["db_id"] == "mock_sales"
        assert task.metadata["difficulty"] == "simple"
        assert task.metadata["golden_sql"] == "SELECT price FROM products WHERE name = 'Server Node'"
        assert task.db_path is not None
        assert os.path.exists(task.db_path)

    def test_get_task_and_manifest(self, mock_bird_sql_dir):
        loader = BirdSqlLoader(dataset_root=mock_bird_sql_dir, split=TaskSplit.DEV)
        task = loader.get_task("bird_mock_01")
        assert task.query == "What is the price of the Server Node?"

        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "bird_sql"
        assert manifest["total_tasks"] == 1
        assert manifest["difficulty_breakdown"]["simple"] == 1
        assert manifest["database_count"] == 1

    def test_load_from_in_memory_tasks(self):
        tasks_data = [
            {
                "question_id": "mem_01",
                "db_id": "test_db",
                "question": "How many rows are in table T?",
                "SQL": "SELECT count(*) FROM T;",
                "difficulty": "challenging"
            }
        ]
        loader = BirdSqlLoader(tasks_data=tasks_data)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "mem_01"
        assert tasks[0].metadata["difficulty"] == "challenging"
