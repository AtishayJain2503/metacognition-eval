"""
tests.unit.test_tools.test_tabular
----------------------------------
Unit tests for tabular data engine (CSV/Parquet ingestion, profiling, sampling, and SQLite bridge).
"""

import os
import sqlite3
import pandas as pd
import pytest

from nemo_eval.tools.tabular import TabularEngine


class TestTabularEngine:
    """Tests for multi-format tabular ingestion, profiling, sampling, and SQLite bridging."""

    def test_csv_delimiters_auto_detection(self, tmp_path):
        delimiters = [",", "\t", ";", "|"]
        for delim in delimiters:
            file_path = tmp_path / f"test_delim_{ord(delim)}.csv"
            df_orig = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
            df_orig.to_csv(file_path, sep=delim, index=False)

            df_loaded = TabularEngine.load_dataset(str(file_path))
            assert df_loaded.shape == (3, 2)
            assert list(df_loaded.columns) == ["col_a", "col_b"]
            assert df_loaded["col_a"].tolist() == [1, 2, 3]

    def test_parquet_loading(self, sample_parquet_file):
        df = TabularEngine.load_dataset(sample_parquet_file)
        assert df.shape == (50, 3)
        assert "record_id" in df.columns
        assert "metric_val" in df.columns

    def test_inspect_schema(self, sample_csv_file):
        schema_info = TabularEngine.inspect_schema(sample_csv_file)
        assert schema_info.file_format == "csv"
        assert schema_info.shape == {"rows": 20, "columns": 5}
        assert schema_info.file_size_bytes > 0
        assert "B" in schema_info.memory_usage_human or "KB" in schema_info.memory_usage_human

        col_dict = {c["name"]: c for c in schema_info.columns}
        assert "price" in col_dict
        assert "float" in col_dict["price"]["dtype"] or "int" in col_dict["price"]["dtype"]
        assert col_dict["notes"]["null_count"] == 8
        assert col_dict["notes"]["null_percentage"] == 40.0

    def test_profile_summary(self, sample_csv_file):
        summary = TabularEngine.profile_summary(sample_csv_file)
        assert summary.shape == {"rows": 20, "columns": 5}

        # Numeric column stats
        price_stat = summary.column_summaries["price"]
        assert price_stat.mean is not None
        assert price_stat.min == 10.5
        assert price_stat.max == 45.0

        # Categorical column stats
        cat_stat = summary.column_summaries["category"]
        assert cat_stat.unique_count == 3
        assert cat_stat.top_value in ["A", "B"]
        assert len(cat_stat.sample_values) == 3

    def test_get_sample_head_and_tail(self, sample_csv_file):
        # Head sample
        head_sample = TabularEngine.get_sample(sample_csv_file, action="head", n_rows=3)
        assert head_sample.action == "head"
        assert head_sample.n_rows_returned == 3
        assert len(head_sample.records) == 3
        assert "id" in head_sample.records[0]
        assert "|" in head_sample.markdown_table  # Markdown table contains pipes

        # Tail sample
        tail_sample = TabularEngine.get_sample(sample_csv_file, action="tail", n_rows=2)
        assert tail_sample.action == "tail"
        assert tail_sample.n_rows_returned == 2
        assert len(tail_sample.records) == 2

    def test_tabular_to_sqlite_bridge(self, sample_csv_file):
        conn = sqlite3.connect(":memory:")
        rows_loaded = TabularEngine.load_to_sqlite(sample_csv_file, "dataset_table", conn)
        assert rows_loaded == 20

        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM dataset_table WHERE category = 'A';")
        assert cursor.fetchone()[0] == 8

        cursor.execute("SELECT AVG(price) FROM dataset_table;")
        avg_price = cursor.fetchone()[0]
        assert avg_price > 0.0
        conn.close()

    def test_inspect_tool_wrapper(self, sample_csv_file):
        # Schema action
        res_schema = TabularEngine.inspect_tool(sample_csv_file, action="schema")
        assert res_schema.status == "success"
        assert res_schema.data["shape"]["rows"] == 20

        # Summary action
        res_summary = TabularEngine.inspect_tool(sample_csv_file, action="summary")
        assert res_summary.status == "success"
        assert "price" in res_summary.data["column_summaries"]

        # Missing file error
        res_missing = TabularEngine.inspect_tool("nonexistent_file.csv", action="schema")
        assert res_missing.status == "error"
        assert res_missing.error is not None
        assert res_missing.error.error_type == "FileNotFoundError"
