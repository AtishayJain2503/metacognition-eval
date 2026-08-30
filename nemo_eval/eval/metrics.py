"""
nemo_eval.eval.metrics
======================
Statistical aggregate metrics, unbiased Pass@k estimator, execution accuracy (EX),
and evaluation scorecard generation.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Union

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.base import EvalResult


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased Pass@k accuracy estimator:
    Pass@k = 1.0 if (n - c) < k else 1.0 - (comb(n - c, k) / comb(n, k))
    
    Args:
        n: Total number of generated samples per task (n >= k)
        c: Number of correct / passing samples (0 <= c <= n)
        k: Sample subset evaluation size (k >= 1)
    """
    if k < 1:
        raise ValueError(f"k must be greater than or equal to 1, got {k}")
    if n < k:
        raise ValueError(f"Total samples n ({n}) must be greater than or equal to k ({k})")
    if c < 0 or c > n:
        raise ValueError(f"Correct samples c ({c}) must be in range [0, {n}]")

    if c == n:
        return 1.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # Unbiased hypergeometric probability
    comb_fail = math.comb(n - c, k)
    comb_total = math.comb(n, k)
    return 1.0 - (comb_fail / comb_total)


def compute_execution_accuracy(results: List[EvalResult]) -> float:
    """Compute overall execution accuracy (fraction of is_correct == True)."""
    if not results:
        return 0.0
    correct_count = sum(1 for r in results if r.is_correct)
    return correct_count / len(results)


def compute_mean_score(results: List[EvalResult]) -> float:
    """Compute mean normalized correctness score across evaluation results."""
    if not results:
        return 0.0
    total_score = sum(r.score for r in results)
    return total_score / len(results)


def generate_scorecard(
    tasks: List[BenchmarkTask],
    results: List[EvalResult],
    pass_at_k_samples: Optional[Dict[str, List[EvalResult]]] = None,
    k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Generate aggregate evaluation scorecard with breakdowns by benchmark suite,
    difficulty, semantic category, and evaluation type.
    """
    if len(tasks) != len(results):
        raise ValueError(f"Mismatch: {len(tasks)} tasks vs {len(results)} results.")

    total_tasks = len(tasks)
    passed_tasks = sum(1 for r in results if r.is_correct)
    execution_accuracy = passed_tasks / total_tasks if total_tasks > 0 else 0.0
    mean_score = sum(r.score for r in results) / total_tasks if total_tasks > 0 else 0.0

    # Group breakdowns
    by_benchmark: Dict[str, Dict[str, Any]] = {}
    by_eval_type: Dict[str, Dict[str, Any]] = {}
    by_difficulty: Dict[str, Dict[str, Any]] = {}
    by_semantic_type: Dict[str, Dict[str, Any]] = {}

    for task, res in zip(tasks, results):
        bm = task.benchmark_name
        et = task.eval_type
        diff = task.metadata.get("difficulty")
        sem = task.metadata.get("semantic_type")

        # By benchmark
        if bm not in by_benchmark:
            by_benchmark[bm] = {"total": 0, "passed": 0, "scores": []}
        by_benchmark[bm]["total"] += 1
        if res.is_correct:
            by_benchmark[bm]["passed"] += 1
        by_benchmark[bm]["scores"].append(res.score)

        # By eval type
        if et not in by_eval_type:
            by_eval_type[et] = {"total": 0, "passed": 0, "scores": []}
        by_eval_type[et]["total"] += 1
        if res.is_correct:
            by_eval_type[et]["passed"] += 1
        by_eval_type[et]["scores"].append(res.score)

        # By difficulty
        if diff:
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "passed": 0, "scores": []}
            by_difficulty[diff]["total"] += 1
            if res.is_correct:
                by_difficulty[diff]["passed"] += 1
            by_difficulty[diff]["scores"].append(res.score)

        # By semantic type
        if sem:
            if sem not in by_semantic_type:
                by_semantic_type[sem] = {"total": 0, "passed": 0, "scores": []}
            by_semantic_type[sem]["total"] += 1
            if res.is_correct:
                by_semantic_type[sem]["passed"] += 1
            by_semantic_type[sem]["scores"].append(res.score)

    def _summarize_groups(group_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        summary = {}
        for k, v in group_dict.items():
            tot = v["total"]
            pas = v["passed"]
            scores = v["scores"]
            summary[k] = {
                "total": tot,
                "passed": pas,
                "accuracy": round(pas / tot, 4) if tot > 0 else 0.0,
                "mean_score": round(sum(scores) / tot, 4) if tot > 0 else 0.0
            }
        return summary

    # Pass@k calculation if multi-sample dictionary provided
    pass_at_k_metrics: Dict[str, float] = {}
    if pass_at_k_samples and k_values:
        for k in k_values:
            k_scores = []
            for tid, samples in pass_at_k_samples.items():
                n_samples = len(samples)
                if n_samples >= k:
                    c_correct = sum(1 for s in samples if s.is_correct)
                    k_scores.append(estimate_pass_at_k(n=n_samples, c=c_correct, k=k))
            if k_scores:
                pass_at_k_metrics[f"pass@{k}"] = round(sum(k_scores) / len(k_scores), 4)

    return {
        "summary": {
            "total_tasks": total_tasks,
            "passed_tasks": passed_tasks,
            "execution_accuracy": round(execution_accuracy, 4),
            "mean_score": round(mean_score, 4),
            "pass_at_k": pass_at_k_metrics
        },
        "by_benchmark": _summarize_groups(by_benchmark),
        "by_eval_type": _summarize_groups(by_eval_type),
        "by_difficulty": _summarize_groups(by_difficulty),
        "by_semantic_type": _summarize_groups(by_semantic_type)
    }


def format_scorecard_markdown(scorecard: Dict[str, Any]) -> str:
    """Format evaluation scorecard into high-signal Markdown summary table."""
    summary = scorecard.get("summary", {})
    lines = [
        "# NeMo Benchmark Evaluation Scorecard",
        f"- **Total Tasks**: {summary.get('total_tasks', 0)}",
        f"- **Passed Tasks**: {summary.get('passed_tasks', 0)}",
        f"- **Execution Accuracy (EX)**: {summary.get('execution_accuracy', 0.0) * 100:.2f}%",
        f"- **Mean Score**: {summary.get('mean_score', 0.0):.4f}",
    ]

    pass_at_k = summary.get("pass_at_k", {})
    if pass_at_k:
        lines.append("## Pass@k Estimations")
        for k, v in pass_at_k.items():
            lines.append(f"- **{k.upper()}**: {v * 100:.2f}%")

    lines.append("\n## Breakdown by Benchmark Suite")
    lines.append("| Benchmark | Total | Passed | Accuracy | Mean Score |")
    lines.append("|---|---|---|---|---|")
    for bm, stats in scorecard.get("by_benchmark", {}).items():
        lines.append(f"| {bm} | {stats['total']} | {stats['passed']} | {stats['accuracy']*100:.1f}% | {stats['mean_score']:.3f} |")

    lines.append("\n## Breakdown by Evaluation Type")
    lines.append("| Eval Type | Total | Passed | Accuracy | Mean Score |")
    lines.append("|---|---|---|---|---|")
    for et, stats in scorecard.get("by_eval_type", {}).items():
        lines.append(f"| {et} | {stats['total']} | {stats['passed']} | {stats['accuracy']*100:.1f}% | {stats['mean_score']:.3f} |")

    if scorecard.get("by_difficulty"):
        lines.append("\n## Breakdown by Difficulty")
        lines.append("| Difficulty | Total | Passed | Accuracy | Mean Score |")
        lines.append("|---|---|---|---|---|")
        for diff, stats in scorecard.get("by_difficulty", {}).items():
            lines.append(f"| {diff} | {stats['total']} | {stats['passed']} | {stats['accuracy']*100:.1f}% | {stats['mean_score']:.3f} |")

    if scorecard.get("by_semantic_type"):
        lines.append("\n## Breakdown by Semantic Type")
        lines.append("| Semantic Type | Total | Passed | Accuracy | Mean Score |")
        lines.append("|---|---|---|---|---|")
        for sem, stats in scorecard.get("by_semantic_type", {}).items():
            lines.append(f"| {sem} | {stats['total']} | {stats['passed']} | {stats['accuracy']*100:.1f}% | {stats['mean_score']:.3f} |")

    return "\n".join(lines)


def export_scorecard_json(scorecard: Dict[str, Any], filepath: str) -> None:
    """Export scorecard dictionary as structured JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
