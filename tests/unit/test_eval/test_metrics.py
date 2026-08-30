"""
tests.unit.test_eval.test_metrics
=================================
Unit tests for Pass@k unbiased estimator, execution accuracy, and scorecard generation.
"""

import json
import math
import os
import pytest

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.metrics import (
    compute_execution_accuracy,
    compute_mean_score,
    estimate_pass_at_k,
    export_scorecard_json,
    format_scorecard_markdown,
    generate_scorecard,
)


class TestPassAtKEstimator:
    """Test mathematical correctness and edge cases of Pass@k estimator."""

    def test_pass_at_1_equals_c_div_n(self):
        # n = 10, c = 4 -> pass@1 = 0.4
        p1 = estimate_pass_at_k(n=10, c=4, k=1)
        assert pytest.approx(p1, rel=1e-5) == 0.40

    def test_pass_at_k_formula(self):
        # n = 10, c = 4, k = 5
        # 1 - comb(6, 5) / comb(10, 5) = 1 - 6 / 252 = 1 - 0.0238095 = 0.97619
        p5 = estimate_pass_at_k(n=10, c=4, k=5)
        expected = 1.0 - (math.comb(6, 5) / math.comb(10, 5))
        assert pytest.approx(p5, rel=1e-5) == expected

    def test_all_correct_or_none_correct(self):
        assert estimate_pass_at_k(n=5, c=5, k=1) == 1.0
        assert estimate_pass_at_k(n=5, c=5, k=3) == 1.0
        assert estimate_pass_at_k(n=5, c=0, k=1) == 0.0
        assert estimate_pass_at_k(n=5, c=0, k=3) == 0.0

    def test_n_minus_c_less_than_k(self):
        # When failures (n - c) < k, at least one success is guaranteed in any subset of size k
        # n = 10, c = 8, k = 5 -> failures = 2 < 5 -> pass@5 = 1.0
        assert estimate_pass_at_k(n=10, c=8, k=5) == 1.0

    def test_parameter_validation_errors(self):
        with pytest.raises(ValueError, match="k must be greater than or equal to 1"):
            estimate_pass_at_k(n=10, c=4, k=0)

        with pytest.raises(ValueError, match="Total samples n"):
            estimate_pass_at_k(n=3, c=2, k=5)

        with pytest.raises(ValueError, match="Correct samples c"):
            estimate_pass_at_k(n=10, c=15, k=2)


class TestScorecardGenerator:
    """Test generating, formatting, and exporting evaluation scorecards."""

    def test_compute_accuracy_and_mean(self):
        results = [
            EvalResult(score=1.0, is_correct=True, eval_type="exact"),
            EvalResult(score=0.0, is_correct=False, eval_type="exact"),
            EvalResult(score=1.0, is_correct=True, eval_type="exact"),
            EvalResult(score=0.5, is_correct=False, eval_type="exact")
        ]
        assert pytest.approx(compute_execution_accuracy(results), rel=1e-3) == 0.50
        assert pytest.approx(compute_mean_score(results), rel=1e-3) == 0.625

    def test_generate_scorecard_breakdowns(self):
        tasks = [
            BenchmarkTask(task_id="t1", benchmark_name="infiagent", query="q1", ground_truth=1, eval_type="float_tol", metadata={"difficulty": "easy"}),
            BenchmarkTask(task_id="t2", benchmark_name="bird_sql", query="q2", ground_truth=[], eval_type="sql_multiset", metadata={"difficulty": "hard"}),
            BenchmarkTask(task_id="t3", benchmark_name="databench", query="q3", ground_truth=True, eval_type="exact", metadata={"semantic_type": "Boolean"}),
        ]
        results = [
            EvalResult(score=1.0, is_correct=True, eval_type="float_tol"),
            EvalResult(score=1.0, is_correct=True, eval_type="sql_multiset"),
            EvalResult(score=0.0, is_correct=False, eval_type="exact")
        ]
        scorecard = generate_scorecard(tasks, results)
        summary = scorecard["summary"]
        assert summary["total_tasks"] == 3
        assert summary["passed_tasks"] == 2
        assert pytest.approx(summary["execution_accuracy"], rel=1e-3) == 2 / 3

        assert "infiagent" in scorecard["by_benchmark"]
        assert "bird_sql" in scorecard["by_benchmark"]
        assert scorecard["by_benchmark"]["bird_sql"]["accuracy"] == 1.0

    def test_format_and_export_scorecard(self, temp_eval_dir):
        tasks = [
            BenchmarkTask(task_id="t1", benchmark_name="synthetic", query="q1", ground_truth=1, eval_type="exact")
        ]
        results = [
            EvalResult(score=1.0, is_correct=True, eval_type="exact")
        ]
        scorecard = generate_scorecard(tasks, results)
        md = format_scorecard_markdown(scorecard)
        assert "# NeMo Benchmark Evaluation Scorecard" in md
        assert "Total Tasks" in md

        json_path = os.path.join(temp_eval_dir, "scorecard.json")
        export_scorecard_json(scorecard, json_path)
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["summary"]["total_tasks"] == 1
