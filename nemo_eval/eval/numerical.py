"""
nemo_eval.eval.numerical
========================
Dual relative and absolute tolerance numerical ground truth evaluation engine
with NaN/inf handling, percentage conversion, and unit stripping.
"""

import math
import re
import time
from typing import Any, Dict, Optional, Tuple, Union

from nemo_eval.eval.base import EvalResult


def extract_numerical_value(val: Any) -> Optional[float]:
    """
    Extract a clean float from numeric or formatted string inputs.
    
    Handles:
    - Floats and Integers
    - Currency symbols: $, €, £, etc.
    - Commas in numbers: 1,250,000.50
    - Percentage signs: 23.5%
    - Scientific notation: 1.25e-4, -3.8E+2
    - Units: 125.5 kg, 45 ms, 90.0 degrees
    - LaTeX wrappers: \\boxed{42.0}, \\text{100}
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, bool):
        # Do not convert bools to 1.0/0.0 implicitly in numerical engine
        return None

    s = str(val).strip()
    # Strip LaTeX and markdown
    s = re.sub(r"\\(?:boxed|text|textbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"[*`_]", "", s).strip()

    # Strip currency symbols
    s = re.sub(r"[\$€£¥₹元]", "", s)
    # Strip commas from numbers
    s = re.sub(r"(?<=\d),(?=\d{3}(?:[^\d]|$))", "", s)

    # Check for infinity
    if s.lower() in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if s.lower() in ("-inf", "-infinity"):
        return float("-inf")
    if s.lower() == "nan":
        return float("nan")

    # Regex search for numeric pattern (with optional exponent)
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None

    return None


def is_percentage_string(val: Any) -> bool:
    """Check if value string explicitly ends with a percentage sign."""
    if isinstance(val, str) and "%" in val:
        return True
    return False


def check_numerical_tolerance(
    cand: float,
    gold: float,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01
) -> Tuple[bool, float, float]:
    """
    Evaluate whether candidate is within relative or absolute tolerance of gold.
    
    Returns (is_match, abs_diff, rel_diff).
    """
    # 0. None check
    if cand is None or gold is None:
        return False, None, None

    # 1. NaN checks
    if math.isnan(gold):
        return math.isnan(cand), 0.0, 0.0
    if math.isnan(cand):
        return False, float("nan"), float("nan")

    # 2. Inf checks
    if math.isinf(gold) or math.isinf(cand):
        is_match = (cand == gold)
        return is_match, 0.0 if is_match else float("inf"), 0.0 if is_match else float("inf")

    # 3. Signed zero normalization
    if cand == 0.0 and gold == 0.0:
        return True, 0.0, 0.0

    # 4. Compute differences
    abs_diff = abs(cand - gold)
    
    # If gold is zero, relative tolerance is undefined; absolute tolerance governs
    if abs(gold) < 1e-9:
        rel_diff = abs_diff / 1e-9
        is_match = abs_diff <= abs_tol
        return is_match, abs_diff, rel_diff

    rel_diff = abs_diff / abs(gold)
    is_match = (abs_diff <= abs_tol) or (rel_diff <= rel_tol)
    return is_match, abs_diff, rel_diff


def evaluate_numerical(
    candidate: Any,
    gold: Any,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
    allow_percentage_scaling: bool = True
) -> EvalResult:
    """
    Evaluate candidate against gold reference using dual numerical tolerances.
    """
    start_perf = time.perf_counter()

    cand_float = extract_numerical_value(candidate)
    gold_float = extract_numerical_value(gold)

    if cand_float is None or gold_float is None:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type="float_tol",
            candidate_parsed=candidate,
            gold_target=gold,
            diagnostic_message=f"Numerical extraction failed: candidate='{candidate}' (parsed={cand_float}), gold='{gold}' (parsed={gold_float})",
            execution_time_ms=round(elapsed_ms, 3),
            details={"cand_parsed": cand_float, "gold_parsed": gold_float}
        )

    # Standard direct tolerance comparison
    is_match, abs_diff, rel_diff = check_numerical_tolerance(
        cand=cand_float,
        gold=gold_float,
        rel_tol=rel_tol,
        abs_tol=abs_tol
    )

    scaled_applied = False
    # Percentage conversion check (e.g. 0.235 vs 23.5 or 25% vs 0.25)
    if not is_match and allow_percentage_scaling:
        # Case 1: gold is in [0, 1], candidate is in [0, 100]
        if (0.0 <= gold_float <= 1.0) and (abs(cand_float) > 1.0 or is_percentage_string(candidate)):
            scaled_match, s_abs, s_rel = check_numerical_tolerance(
                cand=cand_float,
                gold=gold_float * 100.0,
                rel_tol=rel_tol,
                abs_tol=abs_tol * 100.0
            )
            if scaled_match:
                is_match = True
                scaled_applied = True
                abs_diff = s_abs
                rel_diff = s_rel

        # Case 2: candidate is in [0, 1], gold is in [0, 100]
        elif (0.0 <= cand_float <= 1.0) and (abs(gold_float) > 1.0 or is_percentage_string(gold)):
            scaled_match, s_abs, s_rel = check_numerical_tolerance(
                cand=cand_float * 100.0,
                gold=gold_float,
                rel_tol=rel_tol,
                abs_tol=abs_tol * 100.0
            )
            if scaled_match:
                is_match = True
                scaled_applied = True
                abs_diff = s_abs
                rel_diff = s_rel

    elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
    diag = (
        f"Numerical match passed (abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}{', scaled' if scaled_applied else ''})."
        if is_match
        else f"Numerical tolerance exceeded: candidate={cand_float}, gold={gold_float} (abs_diff={abs_diff:.6f} > {abs_tol}, rel_diff={rel_diff:.6f} > {rel_tol})"
    )

    return EvalResult(
        score=1.0 if is_match else 0.0,
        is_correct=is_match,
        eval_type="float_tol",
        candidate_parsed=cand_float,
        gold_target=gold_float,
        diagnostic_message=diag,
        execution_time_ms=round(elapsed_ms, 3),
        details={
            "cand_float": cand_float,
            "gold_float": gold_float,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
            "percentage_scaling_applied": scaled_applied
        }
    )
