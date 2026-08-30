"""
tests.unit.test_eval.conftest
=============================
Shared fixtures and SQLite database instances for evaluation engine unit tests.
"""

import os
import sqlite3
import tempfile
import pytest


@pytest.fixture
def temp_eval_dir():
    """Temporary directory for evaluation engine test artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def eval_sqlite_db(temp_eval_dir):
    """Creates a sample SQLite database for SQL matching tests."""
    db_path = os.path.join(temp_eval_dir, "eval_test.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, salary REAL, dept TEXT);")
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?);", [
        (1, "Alice", 95000.0, "Engineering"),
        (2, "Bob", 72000.0, "Marketing"),
        (3, "Charlie", 110000.0, "Engineering"),
        (4, "David", 60000.0, "Support"),
        (5, "Emma", 110000.0, "Engineering")
    ])
    
    cursor.execute("CREATE TABLE departments (dept_name TEXT PRIMARY KEY, budget REAL);")
    cursor.executemany("INSERT INTO departments VALUES (?, ?);", [
        ("Engineering", 500000.0),
        ("Marketing", 200000.0),
        ("Support", 150000.0)
    ])
    
    conn.commit()
    conn.close()
    return db_path
