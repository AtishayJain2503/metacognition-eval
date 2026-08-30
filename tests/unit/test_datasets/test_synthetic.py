"""
tests.unit.test_datasets.test_synthetic
=======================================
Unit tests for SyntheticBenchmarkGenerator (3 SQLite DBs, 3 tabular datasets, 50+ tasks).
"""

import os
import sqlite3
import pandas as pd
import pytest

from nemo_eval.datasets.synthetic import SyntheticBenchmarkGenerator


class TestSyntheticSqliteGenerator:
    """Test generation and relational integrity of the 3 SQLite databases."""

    def test_generate_sales_database(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        db_path = os.path.join(temp_dataset_dir, "enterprise_sales.sqlite")
        res_path = generator.generate_sales_db(db_path)
        assert os.path.exists(res_path)

        conn = sqlite3.connect(res_path)
        cursor = conn.cursor()
        
        # Verify tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name ASC;")
        tables = [r[0] for r in cursor.fetchall()]
        expected_tables = ["categories", "customers", "order_items", "orders", "products", "regions", "sales_reps"]
        for et in expected_tables:
            assert et in tables

        # Verify row counts and queries
        cursor.execute("SELECT count(*) FROM products;")
        assert cursor.fetchone()[0] == 8

        cursor.execute("SELECT count(*) FROM orders;")
        assert cursor.fetchone()[0] == 10

        conn.close()

    def test_generate_hospital_database(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        db_path = os.path.join(temp_dataset_dir, "hospital_records.sqlite")
        res_path = generator.generate_hospital_db(db_path)
        assert os.path.exists(res_path)

        conn = sqlite3.connect(res_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        assert "departments" in tables
        assert "doctors" in tables
        assert "patients" in tables
        assert "admissions" in tables
        assert "treatments" in tables

        cursor.execute("SELECT count(*) FROM doctors;")
        assert cursor.fetchone()[0] == 5
        conn.close()

    def test_generate_financial_database(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        db_path = os.path.join(temp_dataset_dir, "financial_ledger.sqlite")
        res_path = generator.generate_finance_db(db_path)
        assert os.path.exists(res_path)

        conn = sqlite3.connect(res_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM accounts;")
        assert cursor.fetchone()[0] == 7
        cursor.execute("SELECT count(*) FROM transactions;")
        assert cursor.fetchone()[0] == 6
        conn.close()


class TestSyntheticTabularGenerator:
    """Test generation of customer churn CSV, telemetry Parquet, and inventory CSV."""

    def test_generate_customer_churn_csv(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        csv_path = os.path.join(temp_dataset_dir, "customer_churn.csv")
        generator.generate_churn_csv(csv_path, n_rows=1000)
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        assert df.shape[0] == 1000
        assert "churn" in df.columns
        assert "monthly_charges" in df.columns
        assert "tenure_months" in df.columns

    def test_generate_sensor_telemetry_parquet(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        parquet_path = os.path.join(temp_dataset_dir, "sensor_telemetry.parquet")
        generator.generate_telemetry_parquet(parquet_path, n_rows=10000)
        assert os.path.exists(parquet_path)

        df = pd.read_parquet(parquet_path)
        assert df.shape[0] == 10000
        assert "temperature_c" in df.columns
        assert "pressure_kpa" in df.columns
        assert "status" in df.columns

    def test_generate_inventory_csv(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        csv_path = os.path.join(temp_dataset_dir, "inventory.csv")
        generator.generate_inventory_csv(csv_path, n_rows=500)
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        assert df.shape[0] == 500
        assert "sku" in df.columns
        assert "unit_price" in df.columns


class TestSyntheticBenchmarkTasks:
    """Test generating 50+ deterministic tasks covering all evaluation types."""

    def test_get_synthetic_benchmark_tasks_count_and_types(self, temp_dataset_dir):
        generator = SyntheticBenchmarkGenerator(seed=42)
        tasks = generator.get_synthetic_benchmark_tasks(temp_dataset_dir)
        
        # Verify 50+ tasks
        assert len(tasks) >= 50

        # Verify representation across all 4 benchmarks
        benchmarks = set(t.benchmark_name for t in tasks)
        assert "bird_sql" in benchmarks
        assert "infiagent" in benchmarks
        assert "databench" in benchmarks
        assert "synthetic" in benchmarks

        # Verify representation across all 4 eval types
        eval_types = set(t.eval_type for t in tasks)
        assert "exact" in eval_types
        assert "float_tol" in eval_types
        assert "sql_multiset" in eval_types
        assert "dataframe_diff" in eval_types

    def test_synthetic_task_determinism(self, temp_dataset_dir):
        """Two runs with the same seed generate identical tasks."""
        gen1 = SyntheticBenchmarkGenerator(seed=42)
        tasks1 = gen1.get_synthetic_benchmark_tasks(temp_dataset_dir)

        gen2 = SyntheticBenchmarkGenerator(seed=42)
        tasks2 = gen2.get_synthetic_benchmark_tasks(temp_dataset_dir)

        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.task_id == t2.task_id
            assert t1.query == t2.query
            assert t1.eval_type == t2.eval_type
            assert t1.ground_truth == t2.ground_truth
