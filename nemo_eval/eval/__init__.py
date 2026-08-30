"""
nemo_eval.eval
==============
Polymorphic ground truth evaluation engines and statistical metric estimators.
"""

from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.dataframe_diff import (
    align_dataframe_columns,
    coerce_to_dataframe,
    compare_cell_values,
    evaluate_dataframe,
)
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.eval.exact import (
    evaluate_exact,
    normalize_boolean,
    normalize_text,
    strip_articles,
    strip_markdown,
)
from nemo_eval.eval.math_eval import (
    SympyMathEvaluator,
    check_algebraic_equivalence,
    check_fraction_equivalence,
    check_set_and_interval_equivalence,
    evaluate_math_expression,
    extract_latex_boxed,
    normalize_latex_expression,
    parse_math_to_sympy,
)
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
)
from nemo_eval.eval.sql_match import (
    evaluate_sql,
    execute_sql_safely,
    extract_sql_from_text,
    has_order_by_clause,
    normalize_row_tuple,
    normalize_sql_cell,
)

__all__ = [
    "EvalResult",
    "evaluate_task_result",
    "evaluate_exact",
    "normalize_text",
    "normalize_boolean",
    "strip_markdown",
    "strip_articles",
    "evaluate_numerical",
    "extract_numerical_value",
    "check_numerical_tolerance",
    "evaluate_sql",
    "execute_sql_safely",
    "extract_sql_from_text",
    "has_order_by_clause",
    "normalize_sql_cell",
    "normalize_row_tuple",
    "evaluate_dataframe",
    "coerce_to_dataframe",
    "align_dataframe_columns",
    "compare_cell_values",
    "estimate_pass_at_k",
    "compute_execution_accuracy",
    "compute_mean_score",
    "generate_scorecard",
    "format_scorecard_markdown",
    "export_scorecard_json",
    "SympyMathEvaluator",
    "evaluate_math_expression",
    "normalize_latex_expression",
    "parse_math_to_sympy",
    "check_algebraic_equivalence",
    "check_fraction_equivalence",
    "check_set_and_interval_equivalence",
    "extract_latex_boxed",
]
