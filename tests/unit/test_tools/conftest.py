"""
tests.unit.test_tools.conftest
------------------------------
Shared pytest fixtures for Milestone 1 hermetic tool unit tests.
"""

import os
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import pytest

from nemo_eval.tools.repl import PythonREPL
from nemo_eval.tools.sqlite_engine import SQLiteEngine, SQLiteEngineConfig


@pytest.fixture
def repl_tool():
    """Provides a fresh PythonREPL instance with auto-cleanup."""
    repl = PythonREPL(default_timeout=5.0)
    yield repl
    repl.close()


@pytest.fixture
def sample_sqlite_db():
    """Provides an in-memory SQLite database populated with relational benchmark tables."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE,
            signup_date TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_amount REAL,
            order_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Alice", "Smith", "alice@example.com", "2024-01-15"),
            (2, "Bob", "Jones", "bob@example.com", "2024-02-20"),
            (3, "Charlie", "Brown", "charlie@example.com", "2024-03-10"),
        ]
    )
    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [
            (101, 1, 150.50, "2024-04-01"),
            (102, 1, 85.00, "2024-04-05"),
            (103, 2, 299.99, "2024-04-12"),
        ]
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def sample_sqlite_engine():
    """Provides an initialized SQLiteEngine populated with customers & orders tables."""
    engine = SQLiteEngine(SQLiteEngineConfig(read_only=False))
    engine.init_from_sql("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        INSERT INTO categories VALUES (1, 'Electronics'), (2, 'Books'), (3, 'Home');
        INSERT INTO products VALUES 
            (1, 'Laptop', 999.99, 1),
            (2, 'Novel', 14.99, 2),
            (3, 'Blender', 49.99, 3),
            (4, 'Headphones', 79.99, 1),
            (5, 'Cookbook', 24.99, 2);
    """)
    yield engine
    engine.close()


@pytest.fixture
def sample_csv_file(tmp_path):
    """Provide path to a temporary CSV dataset with mixed numerical and categorical data."""
    df = pd.DataFrame({
        "id": range(1, 21),
        "category": ["A", "B", "A", "C", "B"] * 4,
        "price": [10.5, 20.0, 15.75, 45.0, 30.25] * 4,
        "discount": [0.1, 0.0, 0.15, 0.2, 0.05] * 4,
        "notes": ["sample", None, "item", "val", None] * 4,
    })
    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_parquet_file(tmp_path):
    """Provide path to a temporary Parquet dataset."""
    df = pd.DataFrame({
        "record_id": range(100, 150),
        "metric_val": [float(i * 1.5) for i in range(50)],
        "group_tag": [f"tag_{i % 3}" for i in range(50)],
    })
    pq_path = tmp_path / "test_dataset.parquet"
    df.to_parquet(pq_path, index=False)
    return str(pq_path)
