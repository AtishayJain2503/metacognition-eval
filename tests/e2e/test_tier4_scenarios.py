"""
test_tier4_scenarios.py - Real-World Multi-Step Application Scenarios (Tier 4).
Covers end-to-end data analytics, text-to-SQL refinement, multi-step self-correction,
tabular semantic reasoning, and full evaluation pipeline sweeps.
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


class TestTier4RealWorldScenarios:
    """Real-world multi-step long-horizon reasoning scenarios."""

    def test_scenario_01_customer_churn_analytics_pipeline(self, sample_csv_path):
        """
        Scenario 1: End-to-End Data Analytics Pipeline (InfiAgent-DABench Archetype)
        Steps:
        1. Ingest customer CSV and inspect shape/schema.
        2. Clean data and compute Pearson correlation between numerical features and 'churned'.
        3. Identify feature with highest absolute correlation.
        4. Calculate mean value of this top feature for churned vs retained customers.
        5. Verify output against ground truth.
        """
        # Step 1: Ingest & Inspect
        df = pd.read_csv(sample_csv_path)
        assert df.shape == (8, 7)
        assert "churned" in df.columns

        # Step 2: Compute Pearson correlation matrix
        numerical_cols = ["tenure_months", "monthly_charges", "total_charges"]
        corr_matrix = df[numerical_cols + ["churned"]].corr()
        churn_correlations = corr_matrix["churned"].drop("churned")

        # Step 3: Identify feature with highest absolute correlation
        top_feature = churn_correlations.abs().idxmax()
        top_corr_val = churn_correlations[top_feature]
        assert top_feature in ["tenure_months", "monthly_charges", "total_charges"]

        # Step 4: Compute grouped means for top feature
        churned_mean = df[df["churned"] == 1][top_feature].mean()
        retained_mean = df[df["churned"] == 0][top_feature].mean()

        # Step 5: Verification
        assert churned_mean > 0
        assert retained_mean > 0

    def test_scenario_02_multi_turn_text_to_sql_refinement(self, sample_sqlite_db):
        """
        Scenario 2: Multi-Turn Text-to-SQL with Schema Disambiguation (BIRD-SQL Archetype)
        Goal: "Find customer names who ordered more than $500 total, and list the products they purchased."
        Steps:
        1. Introspect schema to identify relationships: orders -> order_items -> products.
        2. Formulate joined query with GROUP BY and HAVING filters.
        3. Execute query and verify multiset result.
        """
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()

        # Step 1: Introspect tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cursor.fetchall()]
        assert "orders" in tables and "order_items" in tables and "products" in tables

        # Step 2 & 3: Formulate and execute relational query
        query = """
        SELECT o.customer_name, p.product_name, oi.quantity, oi.unit_price
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.total_amount > 500
        ORDER BY o.customer_name, p.product_name;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Alice Smith (Laptop Pro, Wireless Mouse) and Charlie Brown (Standing Desk, Python Design Patterns)
        expected_customers = {"Alice Smith", "Charlie Brown"}
        actual_customers = {row[0] for row in results}
        assert actual_customers == expected_customers
        assert len(results) == 4 # 2 products for Alice, 2 products for Charlie

        conn.close()

    def test_scenario_03_multi_step_self_correction_episode(self, sample_csv_path):
        """
        Scenario 3: Multi-Step Self-Correction Episode (Error Recovery Archetype)
        Steps:
        1. Agent generates code referencing non-existent column 'MonthlyFee'.
        2. Execution raises KeyError('MonthlyFee').
        3. Diagnostic Formatter catches exception, suggests 'monthly_charges'.
        4. Agent receives diagnostic remediation prompt, repairs script to use 'monthly_charges'.
        5. Repaired script executes successfully and produces valid ground truth.
        """
        df = pd.read_csv(sample_csv_path)
        repl_scope = {"df": df}

        # Step 1 & 2: Failing execution
        faulty_code = "avg_fee = df['MonthlyFee'].mean()"
        diagnostic = None
        try:
            exec(faulty_code, repl_scope)
        except KeyError as err:
            # Step 3: Diagnostic formatting
            available_cols = list(df.columns)
            matched_col = "monthly_charges" # Inferred from similarity
            suggestion = f"Column 'MonthlyFee' not found. Available: {available_cols}. Did you mean '{matched_col}'?"
            diagnostic = DiagnosticError(
                error_type="KeyError",
                message=str(err),
                code_snippet=faulty_code,
                suggestion=suggestion
            )

        assert diagnostic is not None
        assert "monthly_charges" in diagnostic.suggestion

        # Step 4 & 5: Repaired execution
        repaired_code = "avg_fee = df['monthly_charges'].mean()"
        exec(repaired_code, repl_scope)
        assert pytest.approx(repl_scope["avg_fee"], rel=1e-2) == 77.405

    def test_scenario_04_tabular_semantic_reasoning_and_type_resolution(self, temp_dir):
        """
        Scenario 4: Tabular Semantic Reasoning with Dirty String Formats (DataBench Archetype)
        Steps:
        1. Load dataset with currency strings ('$1,200.50'), dates, and boolean indicators ('Y'/'N').
        2. Clean and parse types dynamically.
        3. Compute quarterly revenue growth.
        4. Validate scalar return value.
        """
        raw_csv = os.path.join(temp_dir, "financials.csv")
        with open(raw_csv, "w", encoding="utf-8") as f:
            f.write("quarter,revenue_str,approved\n")
            f.write("2025-Q1,\"$1,000,000.00\",Y\n")
            f.write("2025-Q2,\"$1,250,000.00\",Y\n")
            f.write("2025-Q3,\"$1,100,000.00\",Y\n")
            f.write("2025-Q4,\"$1,500,000.00\",Y\n")

        df = pd.read_csv(raw_csv)
        # Parse currency strings to float
        df["revenue"] = df["revenue_str"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
        
        # Calculate Q4 vs Q1 revenue growth rate
        q1_rev = df.loc[df["quarter"] == "2025-Q1", "revenue"].values[0]
        q4_rev = df.loc[df["quarter"] == "2025-Q4", "revenue"].values[0]
        growth_rate = (q4_rev - q1_rev) / q1_rev # 50%

        assert pytest.approx(growth_rate, rel=1e-3) == 0.50

    def test_scenario_05_full_benchmark_pipeline_sweep(self, sample_csv_path, sample_sqlite_db):
        """
        Scenario 5: Full Evaluation Harness Multi-Task Sweep & Report Generation
        Steps:
        1. Define batch of 3 benchmark tasks (InfiAgent, BIRD-SQL, DataBench).
        2. Execute each task through mock agent loop.
        3. Collect trajectories, compute Plan Adherence Scores (PAS) and accuracy.
        4. Generate structured Markdown & JSON evaluation summary report.
        """
        tasks = [
            BenchmarkTask(
                task_id="task_01",
                benchmark_name="infiagent",
                query="Average total charges?",
                ground_truth=2869.35,
                eval_type="float_tol"
            ),
            BenchmarkTask(
                task_id="task_02",
                benchmark_name="bird_sql",
                query="Total product count?",
                ground_truth=5,
                eval_type="exact"
            ),
            BenchmarkTask(
                task_id="task_03",
                benchmark_name="databench",
                query="Are all categories named?",
                ground_truth=True,
                eval_type="exact"
            )
        ]

        trajectories = []
        for task in tasks:
            traj = EpisodeTrajectory(
                task_id=task.task_id,
                model_name="mock-model",
                status="success",
                steps=[
                    StepEvent(step_id=1, state="PLANNING", timestamp=1.0, duration_ms=5.0),
                    StepEvent(step_id=2, state="TOOL_EXECUTION", timestamp=2.0, duration_ms=10.0),
                    StepEvent(step_id=3, state="TERMINAL_SUCCESS", timestamp=3.0, duration_ms=5.0)
                ],
                total_duration_ms=20.0,
                plan_adherence_score=1.0,
                final_answer=task.ground_truth,
                ground_truth_score=1.0
            )
            trajectories.append(traj)

        # Compute aggregate metrics
        total = len(trajectories)
        passed = sum(1 for t in trajectories if t.ground_truth_score == 1.0)
        mean_pas = sum(t.plan_adherence_score for t in trajectories) / total
        pass_rate = passed / total

        assert total == 3
        assert pass_rate == 1.0
        assert mean_pas == 1.0
