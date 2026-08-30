"""
tests.unit.test_eval.test_engine
================================
Unit tests for evaluate_task_result polymorphic evaluation router.
"""

import pytest

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.engine import evaluate_task_result


class TestPolymorphicRouter:
    """Test routing BenchmarkTask to appropriate evaluator."""

    def test_route_exact_eval(self):
        task = BenchmarkTask(
            task_id="t_exact",
            benchmark_name="synthetic",
            query="Is today Monday?",
            ground_truth=False,
            eval_type="exact"
        )
        res = evaluate_task_result(task, "No")
        assert res.is_correct is True
        assert res.eval_type == "exact"
        assert res.score == 1.0

    def test_route_float_tol_eval_with_delimiter(self):
        task = BenchmarkTask(
            task_id="t_float",
            benchmark_name="infiagent",
            query="Compute mean score.",
            ground_truth=85.5,
            eval_type="float_tol",
            metadata={"tolerance": 0.01}
        )
        response_text = "Analysis complete.\nFinal Answer: 85.9"
        res = evaluate_task_result(task, response_text)
        assert res.is_correct is True
        assert res.eval_type == "float_tol"

    def test_route_sql_multiset_eval(self, eval_sqlite_db):
        task = BenchmarkTask(
            task_id="t_sql",
            benchmark_name="bird_sql",
            query="Count employees in engineering.",
            db_path=eval_sqlite_db,
            ground_truth=[(3,)],
            eval_type="sql_multiset",
            metadata={"golden_sql": "SELECT count(*) FROM employees WHERE dept = 'Engineering';"}
        )
        cand_sql = "SELECT count(*) FROM employees WHERE dept = 'Engineering';"
        res = evaluate_task_result(task, cand_sql)
        assert res.is_correct is True
        assert res.eval_type == "sql_multiset"

    def test_route_dataframe_diff_eval(self):
        task = BenchmarkTask(
            task_id="t_df",
            benchmark_name="databench",
            query="Summarize table",
            ground_truth=[{"col": 1, "val": 10.0}],
            eval_type="dataframe_diff"
        )
        cand = [{"col": 1, "val": 10.0}]
        res = evaluate_task_result(task, cand)
        assert res.is_correct is True
        assert res.eval_type == "dataframe_diff"

    def test_unknown_eval_type_fallback(self):
        task = BenchmarkTask(
            task_id="t_unknown",
            benchmark_name="synthetic",
            query="Test",
            ground_truth="target",
            eval_type="exact"
        )
        # Force invalid eval_type
        task.eval_type = "non_existent_type"  # type: ignore
        res = evaluate_task_result(task, "target")
        assert res.is_correct is False
        assert "Unknown evaluation strategy" in res.diagnostic_message
