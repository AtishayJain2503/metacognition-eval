"""
conftest.py - Shared fixtures and contracts for NeMo Long-Horizon Evaluation E2E Test Suite.
"""
import io
import os
import sys
import tempfile
import sqlite3
import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Interface Contracts as defined in PROJECT.md
# ---------------------------------------------------------------------------

class DiagnosticError(BaseModel):
    error_type: str
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: str = ""
    raw_traceback: str = ""

class ToolResult(BaseModel):
    status: Literal["success", "error"]
    execution_time_ms: float = 0.0
    data: Any = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[DiagnosticError] = None

class BenchmarkTask(BaseModel):
    task_id: str
    benchmark_name: Literal["infiagent", "bird_sql", "databench", "synthetic"]
    query: str
    context_schema: Optional[Dict[str, Any]] = None
    db_path: Optional[str] = None
    table_path: Optional[str] = None
    ground_truth: Any
    eval_type: Literal["exact", "float_tol", "sql_multiset", "dataframe_diff"]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]

class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Dict[str, Any]] = None

class StepEvent(BaseModel):
    step_id: int
    state: Literal[
        "PLANNING", "ACTION_SELECTION", "TOOL_EXECUTION", "OBSERVATION",
        "VERIFICATION", "SELF_CORRECTION", "FINAL_SYNTHESIS",
        "TERMINAL_SUCCESS", "TERMINAL_FAILURE"
    ]
    timestamp: float
    duration_ms: float
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    output_payload: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)

class EpisodeTrajectory(BaseModel):
    task_id: str
    model_name: str
    status: Literal["success", "failed", "timeout", "max_turns_exceeded"]
    steps: List[StepEvent] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    plan_adherence_score: float = 1.0
    self_correction_attempts: int = 0
    self_correction_success: bool = False
    final_answer: Any = None
    ground_truth_score: float = 0.0


# ---------------------------------------------------------------------------
# Test Fixtures: Synthetic Data & Database Environments
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Provides an isolated temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield td

@pytest.fixture
def sample_csv_path(temp_dir):
    """Creates a deterministic synthetic customer churn CSV for tabular/REPL tests."""
    csv_file = os.path.join(temp_dir, "customers.csv")
    df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"],
        "tenure_months": [12, 24, 6, 36, 48, 2, 18, 60],
        "monthly_charges": [59.99, 89.50, 29.99, 119.00, 75.25, 45.00, 95.50, 105.00],
        "total_charges": [719.88, 2148.00, 179.94, 4284.00, 3612.00, 90.00, 1719.00, 6300.00],
        "contract": ["month-to-month", "one-year", "month-to-month", "two-year", "two-year", "month-to-month", "one-year", "two-year"],
        "churned": [1, 0, 1, 0, 0, 1, 0, 0]
    })
    df.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture
def sample_parquet_path(temp_dir, sample_csv_path):
    """Creates a Parquet version of the synthetic customer churn dataset."""
    parquet_file = os.path.join(temp_dir, "customers.parquet")
    df = pd.read_csv(sample_csv_path)
    df.to_parquet(parquet_file, index=False)
    return parquet_file

@pytest.fixture
def sample_sqlite_db(temp_dir):
    """Creates a deterministic SQLite database populated with relational tables and foreign keys."""
    db_file = os.path.join(temp_dir, "ecommerce.db")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT NOT NULL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category_id INTEGER NOT NULL,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        order_date TEXT NOT NULL,
        total_amount REAL NOT NULL
    );
    """)
    
    cursor.execute("""
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """)
    
    # Seed data
    cursor.executemany("INSERT INTO categories VALUES (?, ?);", [
        (1, "Electronics"),
        (2, "Furniture"),
        (3, "Books")
    ])
    
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?);", [
        (101, "Laptop Pro", 1, 1299.99, 15),
        (102, "Wireless Mouse", 1, 29.99, 120),
        (103, "Ergonomic Desk Chair", 2, 249.50, 30),
        (104, "Standing Desk", 2, 499.00, 10),
        (105, "Python Design Patterns", 3, 45.00, 50)
    ])
    
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?);", [
        (1001, "Alice Smith", "2026-01-15", 1329.98),
        (1002, "Bob Jones", "2026-01-16", 249.50),
        (1003, "Charlie Brown", "2026-01-18", 544.00)
    ])
    
    cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?);", [
        (1, 1001, 101, 1, 1299.99),
        (2, 1001, 102, 1, 29.99),
        (3, 1002, 103, 1, 249.50),
        (4, 1003, 104, 1, 499.00),
        (5, 1003, 105, 1, 45.00)
    ])
    
    conn.commit()
    conn.close()
    return db_file
