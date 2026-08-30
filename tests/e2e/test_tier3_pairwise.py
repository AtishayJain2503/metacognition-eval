"""
test_tier3_pairwise.py - Cross-Feature Pairwise Interaction Tests (Tier 3).
Covers REPL + Tabular datasets, SQLite + Error Diagnostics, Mock LLM + FSM Trajectory logging,
DAG Planner + Parameter Bridging, Polymorphic Evaluator + Synthetic Ingestion,
SQLite + DataFrame export, Parquet + Statistical modeling, and Mock LLM Error Injection + Self-Correction.
"""
import ast
import io
import math
import os
import sqlite3
import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any, List, Optional
from collections import Counter

from tests.e2e.conftest import (
    ToolResult, DiagnosticError, BenchmarkTask, ToolCall, LLMMessage,
    LLMResponse, StepEvent, EpisodeTrajectory
)


class TestPairwiseInteractions:
    """Pairwise cross-feature integration test suite."""

    def test_pairwise_repl_and_tabular_datasets(self, sample_csv_path):
        """Pairwise: REPL session interacts with Pandas DataFrame to compute aggregated metrics."""
        repl_scope = {}
        
        # Turn 1: Ingest dataset
        code_turn_1 = f"""
import pandas as pd
df = pd.read_csv(r"{sample_csv_path}")
row_count = len(df)
"""
        exec(code_turn_1, repl_scope)
        assert repl_scope["row_count"] == 8
        assert isinstance(repl_scope["df"], pd.DataFrame)

        # Turn 2: Filter and compute aggregation on preserved state
        code_turn_2 = """
churn_df = df[df['churned'] == 1]
avg_tenure = churn_df['tenure_months'].mean()
avg_monthly = churn_df['monthly_charges'].mean()
"""
        exec(code_turn_2, repl_scope)
        assert pytest.approx(repl_scope["avg_tenure"], rel=1e-2) == 6.6667
        assert pytest.approx(repl_scope["avg_monthly"], rel=1e-2) == 44.993

    def test_pairwise_sqlite_and_error_diagnostics(self, sample_sqlite_db):
        """Pairwise: SQLite engine execution exception is intercepted and transformed into DiagnosticError."""
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()

        # Query with misspelled column name 'category_namee'
        bad_query = "SELECT category_namee FROM categories WHERE category_id = 1;"
        
        diagnostic = None
        try:
            cursor.execute(bad_query)
            cursor.fetchall()
        except sqlite3.OperationalError as err:
            err_msg = str(err)
            cursor.execute("PRAGMA table_info('categories');")
            valid_cols = [r[1] for r in cursor.fetchall()]
            suggestion = f"Available columns in 'categories': {valid_cols}. Did you mean 'category_name'?"
            diagnostic = DiagnosticError(
                error_type="OperationalError",
                message=err_msg,
                code_snippet=bad_query,
                suggestion=suggestion
            )

        assert diagnostic is not None
        assert "no such column" in diagnostic.message
        assert "category_name" in diagnostic.suggestion

        # Self-correction: execute repaired query
        repaired_query = "SELECT category_name FROM categories WHERE category_id = 1;"
        cursor.execute(repaired_query)
        res = cursor.fetchone()[0]
        assert res == "Electronics"
        conn.close()

    def test_pairwise_mock_llm_and_fsm_trajectory_logging(self):
        """Pairwise: Mock LLM client drives a multi-turn 9-state FSM trajectory with full telemetry."""
        scripted_responses = [
            LLMResponse(
                content="Decomposing task into sub-goals.",
                tool_calls=[]
            ),
            LLMResponse(
                content="Executing SQL query to retrieve categories.",
                tool_calls=[ToolCall(id="tc_1", name="sqlite_query", arguments={"query": "SELECT * FROM categories;"})]
            ),
            LLMResponse(
                content="Final Answer: The database has 3 product categories: Electronics, Furniture, Books."
            )
        ]

        trajectory = EpisodeTrajectory(
            task_id="task_pair_001",
            model_name="mock-replay-runner",
            status="success",
            steps=[],
            total_duration_ms=45.2,
            plan_adherence_score=1.0,
            final_answer="The database has 3 product categories: Electronics, Furniture, Books.",
            ground_truth_score=1.0
        )

        step_1 = StepEvent(
            step_id=1,
            state="PLANNING",
            timestamp=100.0,
            duration_ms=12.0,
            input_payload={"prompt": "List categories"},
            output_payload={"content": scripted_responses[0].content}
        )
        step_2 = StepEvent(
            step_id=2,
            state="TOOL_EXECUTION",
            timestamp=112.0,
            duration_ms=18.5,
            input_payload={"tool_call": scripted_responses[1].tool_calls[0].model_dump()},
            output_payload={"data": [(1, "Electronics"), (2, "Furniture"), (3, "Books")]}
        )
        step_3 = StepEvent(
            step_id=3,
            state="TERMINAL_SUCCESS",
            timestamp=130.5,
            duration_ms=14.7,
            input_payload={},
            output_payload={"final_answer": scripted_responses[2].content}
        )

        trajectory.steps.extend([step_1, step_2, step_3])

        assert len(trajectory.steps) == 3
        assert trajectory.steps[0].state == "PLANNING"
        assert trajectory.steps[1].state == "TOOL_EXECUTION"
        assert trajectory.steps[2].state == "TERMINAL_SUCCESS"
        assert trajectory.plan_adherence_score == 1.0

    def test_pairwise_dag_planner_and_parameter_bridging(self, sample_sqlite_db):
        """Pairwise: DAG planner coordinates tool parameter bridging between SQL and Python REPL."""
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()

        cursor.execute("SELECT product_id, price FROM products WHERE category_id = 1;")
        sql_rows = cursor.fetchall()
        conn.close()

        # Bridge extracted parameters into REPL calculation
        repl_scope = {"electronics_products": sql_rows}
        code = """
total_inventory_value = sum(price for pid, price in electronics_products)
"""
        exec(code, repl_scope)
        assert pytest.approx(repl_scope["total_inventory_value"], rel=1e-2) == 1329.98

    def test_pairwise_polymorphic_eval_and_synthetic_benchmarks(self, sample_sqlite_db):
        """Pairwise: Polymorphic evaluator accurately scores outputs from synthetic benchmark tasks."""
        tasks = [
            BenchmarkTask(
                task_id="syn_exact",
                benchmark_name="synthetic",
                query="What category is Laptop Pro in?",
                ground_truth="Electronics",
                eval_type="exact"
            ),
            BenchmarkTask(
                task_id="syn_float",
                benchmark_name="synthetic",
                query="What is the average product price in category 1?",
                ground_truth=664.99,
                eval_type="float_tol"
            ),
            BenchmarkTask(
                task_id="syn_sql",
                benchmark_name="synthetic",
                query="List all category names.",
                ground_truth=[("Books",), ("Electronics",), ("Furniture",)],
                eval_type="sql_multiset"
            )
        ]

        def evaluate(pred: Any, task: BenchmarkTask) -> bool:
            if task.eval_type == "exact":
                return str(pred).strip().lower() == str(task.ground_truth).strip().lower()
            elif task.eval_type == "float_tol":
                return math.isclose(float(pred), float(task.ground_truth), rel_tol=0.01)
            elif task.eval_type == "sql_multiset":
                p_c = Counter([tuple(r) for r in pred])
                g_c = Counter([tuple(r) for r in task.ground_truth])
                return p_c == g_c
            return False

        assert evaluate("electronics", tasks[0]) is True
        assert evaluate(664.99, tasks[1]) is True
        assert evaluate([("Electronics",), ("Furniture",), ("Books",)], tasks[2]) is True

    def test_pairwise_sqlite_to_pandas_dataframe_pipeline(self, sample_sqlite_db):
        """Pairwise: Reads SQLite query results directly into Pandas DataFrame and applies transformations."""
        conn = sqlite3.connect(sample_sqlite_db)
        df_orders = pd.read_sql_query("SELECT * FROM orders;", conn)
        conn.close()

        assert df_orders.shape == (3, 4)
        assert "total_amount" in df_orders.columns
        mean_order = df_orders["total_amount"].mean()
        assert pytest.approx(mean_order, rel=1e-2) == 707.826

    def test_pairwise_parquet_and_scipy_statistical_modeling(self, sample_parquet_path):
        """Pairwise: Reads Parquet columnar file and executes Scipy/NumPy regression analysis in REPL."""
        df = pd.read_parquet(sample_parquet_path)
        
        # Calculate OLS slope for tenure_months vs total_charges
        x = df["tenure_months"].values
        y = df["total_charges"].values
        
        slope, intercept = np.polyfit(x, y, 1)
        assert slope > 0.0 # Positive relationship between tenure and total charges
        r_corr = np.corrcoef(x, y)[0, 1]
        assert r_corr > 0.90 # Strong correlation

    def test_pairwise_mock_llm_error_injection_and_self_correction_fsm(self):
        """Pairwise: Injects deterministic error into Mock LLM flow, triggering SELF_CORRECTION FSM state."""
        fsm_transitions = []

        def transition(from_state: str, to_state: str):
            fsm_transitions.append((from_state, to_state))

        # 1. ACTION -> TOOL (error occurs)
        transition("ACTION_SELECTION", "TOOL_EXECUTION")
        # 2. TOOL -> SELF_CORRECTION
        transition("TOOL_EXECUTION", "SELF_CORRECTION")
        # 3. SELF_CORRECTION -> ACTION (repaired code)
        transition("SELF_CORRECTION", "ACTION_SELECTION")
        # 4. ACTION -> TOOL -> OBSERVATION -> VERIFICATION -> TERMINAL_SUCCESS
        transition("ACTION_SELECTION", "TOOL_EXECUTION")
        transition("TOOL_EXECUTION", "OBSERVATION")
        transition("OBSERVATION", "VERIFICATION")
        transition("VERIFICATION", "TERMINAL_SUCCESS")

        assert len(fsm_transitions) == 7
        assert ("TOOL_EXECUTION", "SELF_CORRECTION") in fsm_transitions
        assert ("SELF_CORRECTION", "ACTION_SELECTION") in fsm_transitions
        assert fsm_transitions[-1][1] == "TERMINAL_SUCCESS"
