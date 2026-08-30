"""
tests.unit.test_datasets.test_databench
======================================
Unit tests for DataBench 4 semantic type categorization and dataset loader.
"""

import os
import pytest

from nemo_eval.datasets.base import TaskSplit
from nemo_eval.datasets.databench import (
    DataBenchLoader,
    DataBenchSemanticType,
    categorize_semantic_type,
    map_semantic_type_to_eval_strategy,
)


class TestDataBenchCategorization:
    """Test 4 semantic question types classification and eval strategy mapping."""

    def test_categorize_scalar(self):
        assert categorize_semantic_type("What is the average price?", 45.5) == DataBenchSemanticType.SCALAR
        assert categorize_semantic_type("How many orders were placed?", 120) == DataBenchSemanticType.SCALAR
        assert categorize_semantic_type("Explicit scalar", explicit_type="Scalar") == DataBenchSemanticType.SCALAR

    def test_categorize_boolean(self):
        assert categorize_semantic_type("Is item A in stock?", True) == DataBenchSemanticType.BOOLEAN
        assert categorize_semantic_type("Does customer X exist?", "no") == DataBenchSemanticType.BOOLEAN
        assert categorize_semantic_type("Are there any missing values?", explicit_type="Boolean") == DataBenchSemanticType.BOOLEAN

    def test_categorize_list_set(self):
        assert categorize_semantic_type("List all distinct product names.", ["Node", "Array"]) == DataBenchSemanticType.LIST_SET
        assert categorize_semantic_type("Which categories are active?", explicit_type="List/Set") == DataBenchSemanticType.LIST_SET

    def test_categorize_table(self):
        table_records = [{"category": "Servers", "count": 10}, {"category": "Storage", "count": 5}]
        assert categorize_semantic_type("Generate summary table of counts.", table_records) == DataBenchSemanticType.TABLE
        assert categorize_semantic_type("Pivot table by month", explicit_type="Table") == DataBenchSemanticType.TABLE

    def test_map_semantic_type_to_eval_strategy(self):
        assert map_semantic_type_to_eval_strategy(DataBenchSemanticType.SCALAR, 45.2) == "float_tol"
        assert map_semantic_type_to_eval_strategy(DataBenchSemanticType.SCALAR, "California") == "exact"
        assert map_semantic_type_to_eval_strategy(DataBenchSemanticType.BOOLEAN, True) == "exact"
        assert map_semantic_type_to_eval_strategy(DataBenchSemanticType.LIST_SET, ["A", "B"]) == "exact"
        assert map_semantic_type_to_eval_strategy(DataBenchSemanticType.TABLE, [{"a": 1}]) == "dataframe_diff"


class TestDataBenchLoader:
    """Test DataBenchLoader loading from directory and in-memory."""

    def test_load_from_directory(self, mock_databench_dir):
        loader = DataBenchLoader(dataset_root=mock_databench_dir, split=TaskSplit.LITE)
        tasks = loader.load_tasks()
        assert len(tasks) == 3
        
        # Check task 1 (Scalar)
        t1 = tasks[0]
        assert t1.task_id == "db_001"
        assert t1.metadata["semantic_type"] == "Scalar"
        assert t1.eval_type == "float_tol"
        assert t1.table_path is not None
        assert os.path.exists(t1.table_path)

        # Check task 2 (Boolean)
        t2 = tasks[1]
        assert t2.task_id == "db_002"
        assert t2.metadata["semantic_type"] == "Boolean"
        assert t2.eval_type == "exact"

        # Check task 3 (List/Set)
        t3 = tasks[2]
        assert t3.task_id == "db_003"
        assert t3.metadata["semantic_type"] == "List/Set"
        assert t3.eval_type == "exact"

    def test_get_task_and_manifest(self, mock_databench_dir):
        loader = DataBenchLoader(dataset_root=mock_databench_dir, split=TaskSplit.LITE)
        t = loader.get_task("db_001")
        assert t.query == "What is the price of item A?"

        manifest = loader.get_manifest()
        assert manifest["benchmark_name"] == "databench"
        assert manifest["total_tasks"] == 3
        assert manifest["semantic_type_breakdown"]["Scalar"] == 1
        assert manifest["semantic_type_breakdown"]["Boolean"] == 1
        assert manifest["semantic_type_breakdown"]["List/Set"] == 1
