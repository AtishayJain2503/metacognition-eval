"""
tests.unit.test_eval.test_adversarial_m2
========================================
Adversarial stress-testing suite for Milestone 2: Ground Truth Evaluation Engines.
Empirically tests robustness, edge cases, mathematical soundness, and invariants across:
1. Dual relative (eps=0.01) and absolute (delta=0.01) tolerance, zero gold, NaN, inf, percentage scaling, unit stripping.
2. Multiset Counter SQL execution matching, NULL cell normalization, ORDER BY sorting preservation, read-only safety, opcode timeouts.
3. Tabular DataFrame diffing, column reordering, dtype mismatches, missing values / NaN pattern alignment, shape mismatches.
4. Pass@k unbiased estimator mathematical soundness, boundary conditions (k > n, c=0, c=n, n-c < k), scorecard aggregation.
"""

import math
import re
import os
import sqlite3
import tempfile
import time
import numpy as np
import pandas as pd
import pytest

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.dataframe_diff import (
    align_dataframe_columns,
    coerce_to_dataframe,
    compare_cell_values,
    evaluate_dataframe,
)
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.eval.exact import evaluate_exact, normalize_boolean, normalize_text
from nemo_eval.eval.metrics import (
    compute_execution_accuracy,
    compute_mean_score,
    estimate_pass_at_k,
    export_scorecard_json,
    format_scorecard_markdown,
    generate_scorecard,
)
from nemo_eval.eval.numerical import (
    check_numerical_tolerance,
    evaluate_numerical,
    extract_numerical_value,
    is_percentage_string,
)
from nemo_eval.eval.sql_match import (
    evaluate_sql,
    execute_sql_safely,
    extract_sql_from_text,
    has_order_by_clause,
    normalize_row_tuple,
    normalize_sql_cell,
)


# ===========================================================================
# 1. ADVERSARIAL NUMERICAL TOLERANCE & EXTRACTION TESTS
# ===========================================================================

class TestAdversarialNumerical:
    """Stress-tests numerical extraction, dual tolerance boundaries, zero gold, NaN/inf, and percentage scaling."""

    @pytest.mark.parametrize("cand,gold,rel_tol,abs_tol,expected_match", [
        # Exact match
        (100.0, 100.0, 0.01, 0.01, True),
        # Within relative tolerance (1% of 100 is 1.0 -> 99.0 to 101.0)
        (100.99, 100.0, 0.01, 0.001, True),
        (99.01, 100.0, 0.01, 0.001, True),
        (101.0, 100.0, 0.01, 0.001, True),
        (99.0, 100.0, 0.01, 0.001, True),
        # Outside relative tolerance
        (101.01, 100.0, 0.01, 0.001, False),
        (98.99, 100.0, 0.01, 0.001, False),
        # Absolute tolerance dominates for small gold (abs_tol=0.01)
        (0.008, 0.0, 0.01, 0.01, True),
        (-0.008, 0.0, 0.01, 0.01, True),
        (0.010, 0.0, 0.01, 0.01, True),
        (-0.010, 0.0, 0.01, 0.01, True),
        (0.011, 0.0, 0.01, 0.01, False),
        (-0.011, 0.0, 0.01, 0.01, False),
        # Large scale numbers (1e6)
        (1005000.0, 1000000.0, 0.01, 0.01, True),   # 0.5% rel diff
        (1015000.0, 1000000.0, 0.01, 0.01, False),  # 1.5% rel diff
        # Negative numbers
        (-99.5, -100.0, 0.01, 0.01, True),
        (-101.5, -100.0, 0.01, 0.01, False),
    ])
    def test_dual_tolerance_boundaries(self, cand, gold, rel_tol, abs_tol, expected_match):
        """Verify strict adherence to dual relative/absolute tolerance mathematical boundaries."""
        is_match, abs_diff, rel_diff = check_numerical_tolerance(cand, gold, rel_tol=rel_tol, abs_tol=abs_tol)
        assert is_match is expected_match
        eval_res = evaluate_numerical(cand, gold, rel_tol=rel_tol, abs_tol=abs_tol)
        assert eval_res.is_correct is expected_match
        assert eval_res.score == (1.0 if expected_match else 0.0)

    def test_signed_zero_equivalence(self):
        """Verify signed zero +0.0 and -0.0 match unconditionally."""
        res1 = evaluate_numerical(+0.0, -0.0)
        assert res1.is_correct is True
        assert res1.score == 1.0

        res2 = evaluate_numerical("-0.0", "0.0")
        assert res2.is_correct is True

    @pytest.mark.parametrize("cand,gold,expected_match", [
        (float("nan"), float("nan"), True),
        ("nan", "nan", True),
        ("NaN", "NaN", True),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
        (float("nan"), 100.0, False),
        (float("inf"), float("inf"), True),
        (float("-inf"), float("-inf"), True),
        ("inf", "infinity", True),
        ("+inf", "+infinity", True),
        ("-inf", "-infinity", True),
        (float("inf"), float("-inf"), False),
        (float("-inf"), float("inf"), False),
        (1e308, float("inf"), False),
        (float("inf"), 1e308, False),
    ])
    def test_nan_and_inf_handling(self, cand, gold, expected_match):
        """Verify empirical handling of NaN, +inf, and -inf in numerical engine."""
        res = evaluate_numerical(cand, gold)
        assert res.is_correct is expected_match
        assert res.score == (1.0 if expected_match else 0.0)

    @pytest.mark.parametrize("cand,gold,expected_match", [
        # Standard positive percentage strings
        ("23.5%", 0.235, True),
        ("0.235", "23.5%", True),
        ("25%", 0.25, True),
        ("0.25", "25%", True),
        # Direct numeric ratio scaling
        (25.0, 0.25, True),
        (0.25, 25.0, True),
        (0.50, 50.0, True),
        (100.0, 1.0, True),
        ("100%", 1.0, True),
        ("0%", 0.0, True),
        ("0.0%", 0.0, True),
        # Small percentages
        ("0.1%", 0.001, True),
        # Mismatched percentages
        ("30%", 0.25, False),
        ("75%", 0.50, False),
    ])
    def test_percentage_bidirectional_scaling(self, cand, gold, expected_match):
        """Verify bidirectional percentage scaling (0.25 <-> 25% or 0.25 <-> 25.0)."""
        res = evaluate_numerical(cand, gold, allow_percentage_scaling=True)
        assert res.is_correct is expected_match

    @pytest.mark.parametrize("raw_input,expected_val", [
        ("$1,250,000.50", 1250000.50),
        ("€ 45.50", 45.50),
        ("£ 100", 100.0),
        ("¥ 999", 999.0),
        ("₹ 500", 500.0),
        ("元 100", 100.0),
        ("125.5 kg", 125.5),
        ("45 ms", 45.0),
        ("90.0 degrees", 90.0),
        ("\\boxed{42.0}", 42.0),
        ("\\text{Score: 88.0}", 88.0),
        ("\\textbf{100}", 100.0),
        ("**100.5**", 100.5),
        ("*12.3*", 12.3),
        ("_55.2_", 55.2),
        ("`42.0`", 42.0),
        ("1.25e-4", 0.000125),
        ("-3.8E+2", -380.0),
    ])
    def test_unit_currency_latex_and_markdown_stripping(self, raw_input, expected_val):
        """Verify robust regex extraction of numbers from currency, units, LaTeX, and markdown."""
        val = extract_numerical_value(raw_input)
        assert val is not None
        assert pytest.approx(val, rel=1e-5) == expected_val

    @pytest.mark.parametrize("invalid_val", [
        None,
        "N/A",
        "invalid_text",
        "no numbers here",
    ])
    def test_non_numeric_strings_and_none_rejected(self, invalid_val):
        """Empirically assert None and non-numeric strings are rejected by numerical extractor."""
        val = extract_numerical_value(invalid_val)
        assert val is None

        res = evaluate_numerical(invalid_val, 100.0)
        assert res.is_correct is False
        assert res.score == 0.0
        assert "Numerical extraction failed" in res.diagnostic_message

    def test_subclass_bool_type_inspection_edge_case(self):
        """
        Document empirical observation:
        In Python, bool is a subclass of int (issubclass(bool, int) is True).
        Since isinstance(val, (int, float)) is placed before isinstance(val, bool)
        in extract_numerical_value, extract_numerical_value(True) returns 1.0.
        """
        val_true = extract_numerical_value(True)
        val_false = extract_numerical_value(False)
        assert val_true == 1.0
        assert val_false == 0.0

    def test_extreme_subnormal_and_overflow_floats(self):
        """Test subnormal numbers (1e-300) and high-magnitude numbers (1e300)."""
        # Subnormal relative tolerance
        res_sub = evaluate_numerical(1.005e-300, 1.000e-300, rel_tol=0.01)
        assert res_sub.is_correct is True

        # High-magnitude relative tolerance
        res_high = evaluate_numerical(1.005e300, 1.000e300, rel_tol=0.01)
        assert res_high.is_correct is True


# ===========================================================================
# 2. ADVERSARIAL SQL MATCHING & SANDBOX TESTS
# ===========================================================================

class TestAdversarialSqlMatch:
    """Stress-tests Multiset Counter SQL execution, NULL cell normalization, ORDER BY enforcement, and sandbox safety."""

    def test_null_cell_normalization_variants(self):
        """Verify all NULL variants (None, np.nan, 'NULL', 'null', 'None', 'none') map to canonical None."""
        assert normalize_sql_cell(None) is None
        assert normalize_sql_cell(np.nan) is None
        assert normalize_sql_cell(float("nan")) is None
        assert normalize_sql_cell("NULL") is None
        assert normalize_sql_cell("null") is None
        assert normalize_sql_cell("None") is None
        assert normalize_sql_cell("none") is None
        assert normalize_sql_cell("   NULL   ") is None

    def test_numeric_and_string_cell_normalization(self):
        """Verify numeric cell normalization (int/float precision rounding) and string folding."""
        assert normalize_sql_cell(1.0) == 1
        assert normalize_sql_cell("1.0") == 1
        assert normalize_sql_cell(42) == 42
        assert normalize_sql_cell("42") == 42
        # Float rounding to 4 decimals
        assert normalize_sql_cell(10.12344) == 10.1234
        assert normalize_sql_cell(10.12341) == 10.1234
        # String case folding and trimming
        assert normalize_sql_cell("  Engineering  ") == "engineering"
        assert normalize_sql_cell("ALICE") == "alice"

    def test_multiset_row_permutations_match(self):
        """Verify unordered rows match via Multiset Counter when check_order=False."""
        cand_rows = [("Alice", 95000), ("Bob", 72000), ("Charlie", 110000)]
        gold_rows = [("Charlie", 110000), ("Alice", 95000), ("Bob", 72000)]
        res = evaluate_sql(cand_rows, gold_rows, check_order=False)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_multiset_duplicate_counts_strictly_enforced(self):
        """Verify duplicate row counts must match exactly (Counter equivalence)."""
        cand_rows = [("Engineering",), ("Engineering",), ("Marketing",)]
        gold_rows = [("Engineering",), ("Marketing",), ("Marketing",)]
        res = evaluate_sql(cand_rows, gold_rows, check_order=False)
        assert res.is_correct is False
        assert res.score == 0.0

    def test_order_by_detection_and_enforcement(self, eval_sqlite_db):
        """Verify top-level ORDER BY is auto-detected and ordering is strictly enforced."""
        # Top-level ORDER BY
        assert has_order_by_clause("SELECT name FROM employees ORDER BY salary DESC") is True
        assert has_order_by_clause("select name from employees order   by salary asc") is True
        assert has_order_by_clause("SELECT name FROM employees WHERE id = 1") is False

        cand_sql = "SELECT name FROM employees ORDER BY salary ASC;"
        gold_sql = "SELECT name FROM employees ORDER BY salary DESC;"
        # Auto-enforces ordering because gold_sql has ORDER BY
        res = evaluate_sql(cand_sql, gold_sql, db_path=eval_sqlite_db)
        assert res.is_correct is False
        assert "Ordered SQL mismatch" in res.diagnostic_message

        # Explicit override check_order=False ignores row order even if ORDER BY is present
        res_override = evaluate_sql(cand_sql, gold_sql, db_path=eval_sqlite_db, check_order=False)
        assert res_override.is_correct is True

    def test_column_projection_width_mismatch(self):
        """Verify projection width mismatch fails with informative diagnostic."""
        cand_res = [("Alice", 95000, "Engineering")]
        gold_res = [("Alice", 95000)]
        res = evaluate_sql(cand_res, gold_res)
        assert res.is_correct is False
        assert "Column count mismatch" in res.diagnostic_message

    def test_markdown_code_fence_sql_extraction(self):
        """Verify extract_sql_from_text handles backticks and trailing semicolons."""
        md_query = """```sql
        SELECT dept, COUNT(*)
        FROM employees
        GROUP BY dept;
        ```"""
        extracted = extract_sql_from_text(md_query)
        assert extracted.startswith("SELECT dept, COUNT(*)")
        assert not extracted.endswith(";")
        assert "```" not in extracted

    @pytest.mark.parametrize("dml_statement", [
        "DROP TABLE employees;",
        "DELETE FROM employees WHERE id = 1;",
        "UPDATE employees SET salary = 999999;",
        "INSERT INTO employees VALUES (99, 'Hacker', 0, 'Evil');",
        "ALTER TABLE employees ADD COLUMN leaked TEXT;",
        "ATTACH DATABASE 'evil.db' AS evil;",
        "DETACH DATABASE evil;",
        "REINDEX employees;",
    ])
    def test_sandbox_blocks_prohibited_dml_statements(self, eval_sqlite_db, dml_statement):
        """Empirically assert DML mutation statements are blocked by execute_sql_safely."""
        rows, cols, err = execute_sql_safely(eval_sqlite_db, dml_statement)
        assert rows is None
        assert err is not None
        assert "Prohibited write statement" in err or "OperationalError" in err

    def test_runaway_recursive_query_opcode_timeout(self, eval_sqlite_db):
        """Empirically assert infinite recursive CTE query terminates within opcode progress timeout."""
        runaway_query = "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT count(*) FROM cnt;"
        t0 = time.perf_counter()
        rows, cols, err = execute_sql_safely(eval_sqlite_db, runaway_query, timeout_seconds=1.0)
        elapsed = time.perf_counter() - t0
        assert rows is None
        assert err is not None
        assert "timed out" in err.lower() or "operationalerror" in err.lower()
        assert elapsed <= 2.5


# ===========================================================================
# 3. ADVERSARIAL TABULAR DATAFRAME DIFFING TESTS
# ===========================================================================

class TestAdversarialDataFrameDiff:
    """Stress-tests tabular DataFrame diffing, column alignment, dtype coercion, and missing values."""

    def test_column_reordering_alignment(self):
        """Verify candidate DataFrame with shuffled columns is aligned to gold columns."""
        df_cand = pd.DataFrame({"col_b": [20, 40], "col_a": [10, 30], "col_c": [100, 200]})
        df_gold = pd.DataFrame({"col_a": [10, 30], "col_b": [20, 40], "col_c": [100, 200]})
        res = evaluate_dataframe(df_cand, df_gold)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_column_name_case_and_whitespace_trimming(self):
        """Verify column names with leading/trailing whitespace and differing case match."""
        df_cand = pd.DataFrame({"  Sales_Amount  ": [100.0, 200.0], "REGION": ["North", "South"]})
        df_gold = pd.DataFrame({"sales_amount": [100.0, 200.0], "region": ["North", "South"]})
        res = evaluate_dataframe(df_cand, df_gold)
        assert res.is_correct is True

    def test_positional_column_alignment_fallback(self):
        """Verify positional column alignment when column names differ but count matches."""
        df_cand = pd.DataFrame({"predicted_val": [1, 2, 3]})
        df_gold = pd.DataFrame({"actual_val": [1, 2, 3]})
        res = evaluate_dataframe(df_cand, df_gold)
        assert res.is_correct is True

    def test_dtype_mismatches_int_vs_float_and_string_numerics(self):
        """Verify integer vs float and string representations of numbers compare numerically."""
        # Int vs Float
        df_int = pd.DataFrame({"val": [1, 2, 3]})
        df_float = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        assert evaluate_dataframe(df_int, df_float).is_correct is True

        # String numeric vs Float
        df_str = pd.DataFrame({"val": ["1.0", "2.0", "3.0"]})
        assert evaluate_dataframe(df_str, df_float).is_correct is True

    def test_datetime_string_vs_timestamp(self):
        """Verify datetime strings match pandas Timestamp objects."""
        df_str_dt = pd.DataFrame({"date": ["2025-01-01", "2025-06-15"]})
        df_ts_dt = pd.DataFrame({"date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-15")]})
        assert evaluate_dataframe(df_str_dt, df_ts_dt).is_correct is True

    def test_nan_and_missing_value_alignment(self):
        """Verify NaN, None, and pd.NA align across candidate and gold DataFrames."""
        df_cand = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [None, "hello", None]})
        df_gold = pd.DataFrame({"a": [1.0, None, 3.0], "b": [np.nan, "hello", pd.NA]})
        res = evaluate_dataframe(df_cand, df_gold)
        assert res.is_correct is True

        # Mismatched NaN vs non-null
        df_mismatch = pd.DataFrame({"a": [1.0, 999.0, 3.0], "b": [None, "hello", None]})
        res_fail = evaluate_dataframe(df_mismatch, df_gold)
        assert res_fail.is_correct is False
        assert res_fail.details["diff_count"] > 0

    def test_cell_numerical_tolerance_in_dataframe(self):
        """Verify numerical relative and absolute tolerance in tabular cells."""
        df_cand = pd.DataFrame({"metric": [100.5, 200.8]})
        df_gold = pd.DataFrame({"metric": [100.0, 200.0]})
        # With 1% tolerance: 100.5 is within 1% of 100 (1.0), 200.8 is within 1% of 200 (2.0)
        res_pass = evaluate_dataframe(df_cand, df_gold, rel_tol=0.01)
        assert res_pass.is_correct is True

        # With 0.1% tolerance: out of tolerance
        res_fail = evaluate_dataframe(df_cand, df_gold, rel_tol=0.001, abs_tol=0.001)
        assert res_fail.is_correct is False

    def test_unordered_rows_sorting_in_dataframe(self):
        """Verify unordered rows in DataFrame match when check_order=False."""
        df_cand = pd.DataFrame({"id": [3, 1, 2], "name": ["Charlie", "Alice", "Bob"]})
        df_gold = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        res = evaluate_dataframe(df_cand, df_gold, check_order=False)
        assert res.is_correct is True

    def test_shape_mismatches(self):
        """Verify shape mismatches (different row or column count) fail immediately."""
        df_rows_3 = pd.DataFrame({"id": [1, 2, 3]})
        df_rows_2 = pd.DataFrame({"id": [1, 2]})
        res1 = evaluate_dataframe(df_rows_3, df_rows_2)
        assert res1.is_correct is False
        assert "shape mismatch" in res1.diagnostic_message

        df_cols_2 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        res2 = evaluate_dataframe(df_cols_2, df_rows_2)
        assert res2.is_correct is False
        assert "shape mismatch" in res2.diagnostic_message


# ===========================================================================
# 4. ADVERSARIAL PASS@K ESTIMATOR & SCORECARD METRICS TESTS
# ===========================================================================

class TestAdversarialMetricsPassAtK:
    """Stress-tests unbiased Pass@k estimator mathematical soundness, boundary cases, and scorecard aggregation."""

    @pytest.mark.parametrize("n,c,k,expected", [
        # Pass@1 always equals c / n
        (1, 1, 1, 1.0),
        (1, 0, 1, 0.0),
        (10, 5, 1, 0.5),
        (100, 37, 1, 0.37),
        # c = 0 always yields 0.0
        (5, 0, 1, 0.0),
        (10, 0, 3, 0.0),
        (20, 0, 10, 0.0),
        # c = n always yields 1.0
        (5, 5, 1, 1.0),
        (10, 10, 5, 1.0),
        (20, 20, 10, 1.0),
        # (n - c) < k implies guaranteed pass (at least one correct in any k subset)
        (10, 8, 5, 1.0),   # 10 - 8 = 2 failures < 5 samples
        (10, 9, 3, 1.0),   # 10 - 9 = 1 failure < 3 samples
        (5, 4, 2, 1.0),    # 5 - 4 = 1 failure < 2 samples
    ])
    def test_pass_at_k_exact_boundary_conditions(self, n, c, k, expected):
        """Verify boundary invariants: k=1, c=0, c=n, and (n - c) < k."""
        result = estimate_pass_at_k(n=n, c=c, k=k)
        assert pytest.approx(result, rel=1e-6) == expected

    @pytest.mark.parametrize("n,c,k", [
        (10, 4, 3),
        (15, 7, 5),
        (20, 10, 4),
        (30, 12, 6),
        (50, 25, 10),
        (100, 40, 10),
    ])
    def test_pass_at_k_exact_hypergeometric_formula(self, n, c, k):
        """Verify estimate_pass_at_k matches unbiased combinatorial hypergeometric formula: 1 - comb(n-c, k) / comb(n, k)."""
        calc = estimate_pass_at_k(n=n, c=c, k=k)
        expected = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
        assert pytest.approx(calc, rel=1e-6) == expected
        assert 0.0 <= calc <= 1.0

    @pytest.mark.parametrize("invalid_n,invalid_c,invalid_k,match_err", [
        (10, 4, 0, "k must be greater than or equal to 1"),
        (10, 4, -1, "k must be greater than or equal to 1"),
        (3, 2, 5, "Total samples n"),
        (10, -1, 2, "Correct samples c"),
        (10, 15, 2, "Correct samples c"),
    ])
    def test_pass_at_k_parameter_validation_exceptions(self, invalid_n, invalid_c, invalid_k, match_err):
        """Verify strict parameter validation throws ValueErrors for invalid domains."""
        with pytest.raises(ValueError, match=match_err):
            estimate_pass_at_k(n=invalid_n, c=invalid_c, k=invalid_k)

    def test_scorecard_aggregation_with_multi_sample_pass_at_k(self):
        """Verify scorecard generator correctly integrates multi-sample Pass@k metrics."""
        tasks = [
            BenchmarkTask(task_id="t1", benchmark_name="bird_sql", query="q1", ground_truth=1, eval_type="exact"),
            BenchmarkTask(task_id="t2", benchmark_name="infiagent", query="q2", ground_truth=2, eval_type="float_tol"),
        ]
        results = [
            EvalResult(score=1.0, is_correct=True, eval_type="exact"),
            EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
        ]

        # Multi-sample dictionary: 5 samples per task
        # t1: 3 passed out of 5 -> pass@1 = 3/5 = 0.6, pass@3 = 1 - comb(2,3)/comb(5,3) = 1.0
        # t2: 5 passed out of 5 -> pass@1 = 1.0, pass@3 = 1.0
        pass_at_k_samples = {
            "t1": [
                EvalResult(score=1.0, is_correct=True, eval_type="exact"),
                EvalResult(score=1.0, is_correct=True, eval_type="exact"),
                EvalResult(score=1.0, is_correct=True, eval_type="exact"),
                EvalResult(score=0.0, is_correct=False, eval_type="exact"),
                EvalResult(score=0.0, is_correct=False, eval_type="exact"),
            ],
            "t2": [
                EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
                EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
                EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
                EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
                EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
            ]
        }

        scorecard = generate_scorecard(
            tasks=tasks,
            results=results,
            pass_at_k_samples=pass_at_k_samples,
            k_values=[1, 3]
        )

        pass_metrics = scorecard["summary"]["pass_at_k"]
        # Mean pass@1 across t1 (0.6) and t2 (1.0) is (0.6 + 1.0) / 2 = 0.80
        assert pytest.approx(pass_metrics["pass@1"], rel=1e-3) == 0.80
        # Mean pass@3 across t1 (1.0) and t2 (1.0) is 1.0
        assert pytest.approx(pass_metrics["pass@3"], rel=1e-3) == 1.00


# ===========================================================================
# 5. ADVERSARIAL POLYMORPHIC ENGINE ROUTER TESTS
# ===========================================================================

class TestAdversarialEngineRouter:
    """Stress-tests polymorphic evaluation router across heterogeneous BenchmarkTask specifications."""

    def test_router_exact_boolean_normalization(self):
        task = BenchmarkTask(
            task_id="t_bool",
            benchmark_name="databench",
            query="Is valid?",
            ground_truth=True,
            eval_type="exact"
        )
        res = evaluate_task_result(task, "yes")
        assert res.is_correct is True
        assert res.eval_type == "exact"

    def test_router_float_tolerance_custom_override(self):
        task = BenchmarkTask(
            task_id="t_float",
            benchmark_name="infiagent",
            query="What is 100/3?",
            ground_truth=33.333,
            eval_type="float_tol",
            metadata={"tolerance": 0.05}
        )
        # 33.0 is ~1% diff, within custom 5% tolerance
        res = evaluate_task_result(task, "33.0")
        assert res.is_correct is True

    def test_router_unknown_eval_type_fallback(self):
        task = BenchmarkTask.model_construct(
            task_id="t_unknown",
            benchmark_name="synthetic",
            query="test",
            ground_truth="abc",
            eval_type="unknown_future_strategy",
            metadata={}
        )
        res = evaluate_task_result(task, "abc")
        assert res.is_correct is False
        assert "Unknown evaluation strategy" in res.diagnostic_message
