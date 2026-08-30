"""
nemo_eval.eval.engine
=====================
Polymorphic ground truth evaluation router dispatching across all benchmark task types.
"""

import time
from typing import Any, Dict, Optional, Union

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.datasets.infiagent import extract_final_answer
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.dataframe_diff import evaluate_dataframe
from nemo_eval.eval.exact import evaluate_exact
from nemo_eval.eval.numerical import evaluate_numerical
from nemo_eval.eval.sql_match import evaluate_sql


def evaluate_task_result(
    task: BenchmarkTask,
    candidate_output: Any,
    db_engine: Optional[Any] = None,
    custom_tolerances: Optional[Dict[str, float]] = None
) -> EvalResult:
    """
    Polymorphic evaluation router for BenchmarkTask instances.
    Extracts candidate value if necessary and delegates to the matching evaluation engine.
    """
    start_perf = time.perf_counter()
    eval_type = task.eval_type
    gold = task.ground_truth
    tolerances = custom_tolerances or {}
    rel_tol = tolerances.get("rel_tol", task.metadata.get("tolerance", 0.01))
    abs_tol = tolerances.get("abs_tol", 0.01)

    # Pre-extract final answer string if candidate is a multi-line agent response
    parsed_candidate = candidate_output
    if isinstance(candidate_output, str) and eval_type != "sql_multiset":
        extracted = extract_final_answer(candidate_output)
        if extracted is not None:
            parsed_candidate = extracted

    # Dispatch to specific engine
    if eval_type == "exact":
        return evaluate_exact(
            candidate=parsed_candidate,
            gold=gold,
            ignore_case=True,
            strip_punctuation=True,
            unordered_collection=True
        )

    elif eval_type == "float_tol":
        return evaluate_numerical(
            candidate=parsed_candidate,
            gold=gold,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )

    elif eval_type == "sql_multiset":
        db_path = task.db_path
        if db_engine is not None and hasattr(db_engine, "config") and getattr(db_engine.config, "db_path", None):
            db_path = db_engine.config.db_path

        # If gold is golden_sql string, or metadata has golden_sql
        gold_ref = gold
        if isinstance(gold, list) and "golden_sql" in task.metadata and task.metadata["golden_sql"]:
            gold_ref = task.metadata["golden_sql"]

        return evaluate_sql(
            candidate_sql_or_result=parsed_candidate,
            gold_sql_or_result=gold_ref,
            db_path=db_path,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )

    elif eval_type == "dataframe_diff":
        return evaluate_dataframe(
            candidate_df=parsed_candidate,
            gold_df=gold,
            check_order=False,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )

    elif eval_type in ("math_symbolic", "fraction", "set"):
        from nemo_eval.eval.math_eval import evaluate_math_expression
        return evaluate_math_expression(
            candidate=parsed_candidate,
            gold=gold,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            eval_type=eval_type
        )

    else:
        # Fallback to exact
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type=eval_type,
            candidate_parsed=parsed_candidate,
            gold_target=gold,
            diagnostic_message=f"Unknown evaluation strategy: '{eval_type}'.",
            execution_time_ms=round(elapsed_ms, 3)
        )
