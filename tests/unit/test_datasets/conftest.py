"""
tests.unit.test_datasets.conftest
=================================
Shared test fixtures and temporary mock data for dataset tests.
"""

import json
import os
import sqlite3
import tempfile
import pandas as pd
import pytest


@pytest.fixture
def temp_dataset_dir():
    """Provides an isolated temporary directory for dataset fixtures."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def mock_infiagent_jsonl(temp_dataset_dir):
    """Creates a mock InfiAgent-DABench JSONL file."""
    file_path = os.path.join(temp_dataset_dir, "test.jsonl")
    records = [
        {
            "task_id": "infi_test_001",
            "instruction": "Compute the average monthly charge for customers.",
            "answer": 77.405,
            "question_type": "closed_form",
            "data_path": "customers.csv"
        },
        {
            "task_id": "infi_test_002",
            "instruction": "Transform table to aggregate by contract type.",
            "answer": [{"contract": "one-year", "count": 2}],
            "question_type": "data_transformation",
            "data_path": "customers.csv"
        }
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return file_path


@pytest.fixture
def mock_bird_sql_dir(temp_dataset_dir):
    """Creates a mock BIRD-SQL directory with dev.json, tables.json, and a sqlite database."""
    db_dir = os.path.join(temp_dataset_dir, "databases", "mock_sales")
    os.makedirs(db_dir, exist_ok=True)
    db_file = os.path.join(db_dir, "mock_sales.sqlite")
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE products (product_id INT PRIMARY KEY, name TEXT, price REAL);")
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?);", [
        (1, "Server Node", 1500.0),
        (2, "Storage Array", 3200.0)
    ])
    conn.commit()
    conn.close()

    tables_json = os.path.join(temp_dataset_dir, "tables.json")
    tables_meta = [
        {
            "db_id": "mock_sales",
            "table_names_original": ["products"],
            "column_names_original": [[-1, "*"], [0, "product_id"], [0, "name"], [0, "price"]],
            "column_types": ["text", "number", "text", "number"]
        }
    ]
    with open(tables_json, "w", encoding="utf-8") as f:
        json.dump(tables_meta, f)

    dev_json = os.path.join(temp_dataset_dir, "dev.json")
    tasks = [
        {
            "question_id": "bird_mock_01",
            "db_id": "mock_sales",
            "question": "What is the price of the Server Node?",
            "evidence": "Server Node refers to products.name = 'Server Node'",
            "SQL": "SELECT price FROM products WHERE name = 'Server Node';",
            "difficulty": "simple"
        }
    ]
    with open(dev_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f)

    return temp_dataset_dir


@pytest.fixture
def mock_databench_dir(temp_dataset_dir):
    """Creates a mock DataBench directory with questions.json and a CSV file."""
    csv_file = os.path.join(temp_dataset_dir, "sales.csv")
    df = pd.DataFrame({
        "item": ["A", "B", "C"],
        "price": [10.0, 20.0, 30.0],
        "in_stock": [True, False, True]
    })
    df.to_csv(csv_file, index=False)

    q_file = os.path.join(temp_dataset_dir, "lite.json")
    questions = [
        {
            "task_id": "db_001",
            "question": "What is the price of item A?",
            "answer": 10.0,
            "type": "Scalar",
            "file_path": "sales.csv"
        },
        {
            "task_id": "db_002",
            "question": "Is item B in stock?",
            "answer": False,
            "type": "Boolean",
            "file_path": "sales.csv"
        },
        {
            "task_id": "db_003",
            "question": "List all items.",
            "answer": ["A", "B", "C"],
            "type": "List/Set",
            "file_path": "sales.csv"
        }
    ]
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(questions, f)

    return temp_dataset_dir
