"""
nemo_eval.eval.math_eval
========================
Symbolic mathematical equivalence and polymorphic ground truth evaluator powered by SymPy.

Supports:
- Preprocessing & normalization of LaTeX macros, expressions, and formatting
- Multi-tier symbolic algebra equivalence (simplify, expand, trigsimp, sampling)
- Relative and absolute numerical tolerance (|cand - gold| <= eps + delta * |gold|)
- Rational fractions, multisets, finite sets, and intervals
- Graceful exception recovery and diagnostics
"""

from __future__ import annotations

import math
import re
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from nemo_eval.eval.base import EvalResult


# Standard SymPy parser transformations for implicit multiplication and caret exponentiation
SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def extract_latex_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} expression using balanced brace parsing."""
    if not isinstance(text, str):
        return None
    
    boxed_idx = text.rfind(r"\boxed{")
    if boxed_idx == -1:
        # Also check without backslash
        boxed_idx = text.rfind("boxed{")
        if boxed_idx != -1 and (boxed_idx == 0 or text[boxed_idx - 1] != "\\"):
            prefix_len = len("boxed{")
        else:
            return None
    else:
        prefix_len = len(r"\boxed{")

    start_pos = boxed_idx + prefix_len
    depth = 1
    idx = start_pos
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1

    if depth == 0:
        return text[start_pos:idx - 1].strip()
    return None


def normalize_latex_expression(text: Any) -> str:
    """
    Normalizes LaTeX mathematical strings into valid expressions parseable by SymPy.
    
    Transforms LaTeX commands (\\frac, \\sqrt, \\pi, \\cdot, etc.), trigonometric powers,
    subscripts, factorials, percentages, and formatting tags into standard mathematical syntax.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    s = text.strip()
    if not s:
        return ""

    # 1. Extract boxed expression if present
    boxed = extract_latex_boxed(s)
    if boxed is not None:
        s = boxed.strip()

    # 2. Strip markdown wrappers: code blocks, backticks, bold, italic
    s = re.sub(r'```(?:latex|math|python)?\s*(.*?)\s*```', r'\1', s, flags=re.DOTALL)
    s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)
    s = re.sub(r'__(.*?)__', r'\1', s)
    s = re.sub(r'`(.*?)`', r'\1', s)
    s = s.replace("`", "")

    # Strip dollar signs
    s = s.strip("$").strip()

    # 3. Strip text and formatting macros: \text{...}, \mathrm{...}, \mathbf{...}, \mathit{...}, \operatorname{...}
    for macro in [r"\\text", r"\\mathrm", r"\\mathbf", r"\\mathit", r"\\operatorname", r"\\mbox", r"\\textbf"]:
        for _ in range(5):
            s = re.sub(macro + r'\{([^{}]*)\}', r'\1', s)

    # 4. Strip variable prefixes like "x = ", "y = ", "ans = ", "Answer: ", "Final Answer: "
    s = re.sub(r'^(?:[a-zA-Z]\s*=\s*|ans(?:wer)?\s*[:=]\s*|Final Answer\s*[:=]\s*|The answer is\s*[:=]?\s*)', '', s, flags=re.IGNORECASE).strip()

    # 5. Normalize LaTeX fractions: \frac{a}{b} and \dfrac{a}{b} -> ((a)/(b)) (iterative for nested fractions)
    for _ in range(8):
        s = re.sub(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)

    # 6. Normalize LaTeX roots: \sqrt[n]{x} -> ((x)**(1/(n))), \sqrt{x} -> sqrt(x)
    for _ in range(5):
        s = re.sub(r'\\sqrt\[([^{}]+)\]\{([^{}]+)\}', r'((\2)**(1/(\1)))', s)
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)

    # 7. Normalize trigonometric and transcendental powers: \sin^2(x) -> (sin(x))**2
    trig_names = r"(?:sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|arcsin|arccos|arctan|asin|acos|atan|ln|log|exp)"
    s = re.sub(r'\\?' + f'({trig_names})' + r'\^\{?(\d+)\}?\s*\(([^()]+)\)', r'(\1(\3))**\2', s)
    s = re.sub(r'\\?' + f'({trig_names})' + r'\^\{?(\d+)\}?\s*([a-zA-Z])', r'(\1(\3))**\2', s)

    # Convert standard trig/log function names
    s = re.sub(r'\\ln\b', 'log', s)
    s = re.sub(r'\\log\b', 'log', s)
    s = re.sub(r'\\exp\b', 'exp', s)
    s = re.sub(r'\\(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh)\b', r'\1', s)
    s = re.sub(r'\\(arcsin|asin)\b', 'asin', s)
    s = re.sub(r'\\(arccos|acos)\b', 'acos', s)
    s = re.sub(r'\\(arctan|atan)\b', 'atan', s)

    # 8. Normalize multiplication, division, and operators
    s = s.replace(r"\cdot", " * ").replace(r"\times", " * ").replace(r"\div", " / ")
    s = s.replace(r"\ast", " * ").replace(r"\star", " * ")

    # 9. Normalize mathematical constants
    s = re.sub(r'\\pi\b', 'pi', s)
    s = re.sub(r'\\infty\b', 'oo', s)
    s = re.sub(r'\\pm\b', ' ', s)

    # 10. Normalize factorials: n! -> factorial(n)
    s = re.sub(r'(\d+)!', r'factorial(\1)', s)
    s = re.sub(r'([a-zA-Z])!', r'factorial(\1)', s)

    # 11. Normalize percentages: 25\% or 25% -> ((25)/100)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*(?:\\%|%)', r'((\1)/100)', s)

    # 12. Normalize brackets and delimiters
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    s = s.replace(r"\left\{", "{").replace(r"\right\}", "}")
    s = s.replace(r"\{", "{").replace(r"\}", "}")

    # 13. Exponentiation caret: x^{2} -> x**(2), x^2 -> x**(2)
    s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)
    s = s.replace("^", "**")

    # Clean residual backslashes before letters or numbers
    s = re.sub(r'\\([a-zA-Z])', r'\1', s)
    s = s.replace("\\", "")

    # Clean redundant spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_math_to_sympy(expr_str: Any) -> Optional[sympy.Basic]:
    """
    Safely parses a mathematical string or LaTeX expression into a SymPy object.
    Returns None if parsing fails.
    """
    if expr_str is None:
        return None
    if isinstance(expr_str, (sympy.Basic, sympy.Number, sympy.Symbol)):
        return expr_str
    if isinstance(expr_str, (int, float)):
        return sympy.Number(expr_str)
    if isinstance(expr_str, bool):
        return sympy.Number(1 if expr_str else 0)

    clean_str = normalize_latex_expression(str(expr_str))
    if not clean_str:
        return None

    # First attempt: direct SymPy parser with transformations
    try:
        expr = parse_expr(clean_str, transformations=SYMPY_TRANSFORMATIONS, evaluate=False)
        return expr
    except Exception:
        pass

    # Second attempt: try evaluating via python float or Fraction
    try:
        frac = Fraction(clean_str.replace(" ", ""))
        return sympy.Rational(frac.numerator, frac.denominator)
    except Exception:
        pass

    try:
        num = float(clean_str)
        return sympy.Float(num)
    except Exception:
        pass

    return None


def check_algebraic_equivalence(
    cand_expr: Any,
    gold_expr: Any,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
) -> bool:
    """
    Evaluates whether two SymPy expressions or mathematical candidates are algebraically equivalent.
    """
    # 1. Parse both to SymPy expressions if not already parsed
    cand_sym = cand_expr if isinstance(cand_expr, sympy.Basic) else parse_math_to_sympy(cand_expr)
    gold_sym = gold_expr if isinstance(gold_expr, sympy.Basic) else parse_math_to_sympy(gold_expr)

    if cand_sym is None or gold_sym is None:
        # Fallback string comparison if SymPy parsing failed
        cand_raw = str(cand_expr).strip().lower()
        gold_raw = str(gold_expr).strip().lower()
        return cand_raw == gold_raw if cand_raw and gold_raw else False

    # 2. Exact structural equivalence
    if cand_sym == gold_sym:
        return True

    # 3. Direct algebraic difference simplification: simplify(cand - gold) == 0
    try:
        diff = sympy.simplify(cand_sym - gold_sym)
        if diff == 0 or getattr(diff, "is_zero", False) is True:
            return True
    except Exception:
        pass

    # 4. Expansion comparison: expand(cand) == expand(gold)
    try:
        if sympy.expand(cand_sym) == sympy.expand(gold_sym):
            return True
    except Exception:
        pass

    # 5. Trigonometric simplification: trigsimp(cand - gold) == 0
    try:
        trig_diff = sympy.trigsimp(cand_sym - gold_sym)
        if trig_diff == 0 or getattr(trig_diff, "is_zero", False) is True:
            return True
    except Exception:
        pass

    # 6. Rational factor simplification: factor(cand - gold) == 0
    try:
        if sympy.factor(cand_sym - gold_sym) == 0:
            return True
    except Exception:
        pass

    # 7. Numerical evaluation of difference (constant expressions)
    try:
        if not cand_sym.free_symbols and not gold_sym.free_symbols:
            c_val = complex(sympy.N(cand_sym))
            g_val = complex(sympy.N(gold_sym))
            abs_diff = abs(c_val - g_val)
            g_mag = abs(g_val)
            if abs_diff <= abs_tol or (g_mag > 0 and abs_diff <= rel_tol * g_mag):
                return True
    except Exception:
        pass

    # 8. Free variable sampling for transcendental or complex polynomials
    try:
        free_syms = list(cand_sym.free_symbols.union(gold_sym.free_symbols))
        if free_syms:
            # Test at 5 pseudo-random non-integer sample points
            sample_points = [
                {sym: val for sym, val in zip(free_syms, [1.37 + i * 0.71 for i in range(len(free_syms))])},
                {sym: val for sym, val in zip(free_syms, [2.81 + i * 0.53 for i in range(len(free_syms))])},
                {sym: val for sym, val in zip(free_syms, [3.19 + i * 0.47 for i in range(len(free_syms))])},
                {sym: val for sym, val in zip(free_syms, [4.63 + i * 0.39 for i in range(len(free_syms))])},
                {sym: val for sym, val in zip(free_syms, [5.27 + i * 0.61 for i in range(len(free_syms))])},
            ]
            all_samples_match = True
            for point in sample_points:
                c_eval = cand_sym.subs(point).evalf()
                g_eval = gold_sym.subs(point).evalf()
                c_num = complex(c_eval)
                g_num = complex(g_eval)
                diff_val = abs(c_num - g_num)
                g_mag = abs(g_num)
                if not (diff_val <= abs_tol or (g_mag > 0 and diff_val <= rel_tol * g_mag)):
                    all_samples_match = False
                    break
            if all_samples_match:
                return True
    except Exception:
        pass

    return False


def check_fraction_equivalence(cand_str: Any, gold_str: Any, rel_tol: float = 0.01) -> bool:
    """Check equivalence between fractions, decimals, and rational expressions."""
    c_clean = normalize_latex_expression(cand_str)
    g_clean = normalize_latex_expression(gold_str)

    if c_clean == g_clean:
        return True

    # Try Fraction conversion
    try:
        f_c = Fraction(c_clean.replace(" ", ""))
        f_g = Fraction(g_clean.replace(" ", ""))
        if f_c == f_g:
            return True
    except Exception:
        pass

    # Try SymPy rational
    try:
        expr_c = parse_math_to_sympy(c_clean)
        expr_g = parse_math_to_sympy(g_clean)
        if expr_c is not None and expr_g is not None:
            if check_algebraic_equivalence(expr_c, expr_g, rel_tol=rel_tol):
                return True
    except Exception:
        pass

    # Float fallback
    try:
        val_c = float(eval(c_clean, {"__builtins__": {}}, {}))
        val_g = float(eval(g_clean, {"__builtins__": {}}, {}))
        if abs(val_c - val_g) <= 1e-4 or abs(val_c - val_g) <= rel_tol * abs(val_g):
            return True
    except Exception:
        pass

    return False


def check_set_and_interval_equivalence(
    cand_str: Any,
    gold_str: Any,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
) -> bool:
    """
    Evaluates equivalence of mathematical sets (e.g. {1, 2} == {2, 1}) or intervals.
    """
    c_raw = str(cand_str).strip()
    g_raw = str(gold_str).strip()

    if c_raw == g_raw:
        return True

    def extract_elements(s: str) -> List[str]:
        s_clean = normalize_latex_expression(s).strip()
        # Strip surrounding braces or brackets if set notation
        if (s_clean.startswith("{") and s_clean.endswith("}")) or (s_clean.startswith("[") and s_clean.endswith("]")):
            inner = s_clean[1:-1].strip()
        else:
            inner = s_clean
        if not inner:
            return []
        return [elem.strip() for elem in inner.split(",") if elem.strip()]

    c_elems = extract_elements(c_raw)
    g_elems = extract_elements(g_raw)

    if len(c_elems) != len(g_elems):
        return False

    # Check if elements match bijectively (multiset matching with symbolic equivalence)
    matched_indices: Set[int] = set()
    for c_elem in c_elems:
        found_match = False
        c_sym = parse_math_to_sympy(c_elem)
        for j, g_elem in enumerate(g_elems):
            if j in matched_indices:
                continue
            g_sym = parse_math_to_sympy(g_elem)
            if c_sym is not None and g_sym is not None:
                if check_algebraic_equivalence(c_sym, g_sym, rel_tol=rel_tol, abs_tol=abs_tol):
                    matched_indices.add(j)
                    found_match = True
                    break
            elif c_elem.strip().lower() == g_elem.strip().lower():
                matched_indices.add(j)
                found_match = True
                break
        if not found_match:
            return False

    return len(matched_indices) == len(g_elems)


def evaluate_math_expression(
    candidate: Any,
    gold: Any,
    rel_tol: float = 0.01,
    abs_tol: float = 0.01,
    eval_type: str = "math_symbolic",
) -> EvalResult:
    """
    Master polymorphic mathematical evaluation function.
    
    Returns structured EvalResult with binary verdict, score (0.0 or 1.0),
    parsed outputs, diagnostic message, and execution timing.
    """
    start_time = time.perf_counter()

    if candidate is None or gold is None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type=eval_type,
            candidate_parsed=candidate,
            gold_target=gold,
            diagnostic_message="Candidate or ground truth is None.",
            execution_time_ms=round(elapsed_ms, 3),
        )

    cand_str = str(candidate).strip()
    gold_str = str(gold).strip()

    if not cand_str:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return EvalResult(
            score=0.0,
            is_correct=False,
            eval_type=eval_type,
            candidate_parsed="",
            gold_target=gold_str,
            diagnostic_message="Candidate is empty.",
            execution_time_ms=round(elapsed_ms, 3),
        )

    # 1. Exact string match short-circuit
    if cand_str == gold_str or cand_str.lower() == gold_str.lower():
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return EvalResult(
            score=1.0,
            is_correct=True,
            eval_type=eval_type,
            candidate_parsed=cand_str,
            gold_target=gold_str,
            diagnostic_message="Exact string match.",
            execution_time_ms=round(elapsed_ms, 3),
        )

    # 2. Extract boxed values if present
    cand_unboxed = extract_latex_boxed(cand_str) or cand_str
    gold_unboxed = extract_latex_boxed(gold_str) or gold_str

    if cand_unboxed.strip() == gold_unboxed.strip():
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return EvalResult(
            score=1.0,
            is_correct=True,
            eval_type=eval_type,
            candidate_parsed=cand_unboxed,
            gold_target=gold_unboxed,
            diagnostic_message="Unboxed exact match.",
            execution_time_ms=round(elapsed_ms, 3),
        )

    is_correct = False
    diagnostic = ""

    try:
        if eval_type == "set":
            is_correct = check_set_and_interval_equivalence(
                cand_unboxed, gold_unboxed, rel_tol=rel_tol, abs_tol=abs_tol
            )
            diagnostic = "Set equivalence match." if is_correct else f"Set mismatch: '{cand_str}' vs '{gold_str}'."

        elif eval_type == "fraction":
            is_correct = check_fraction_equivalence(cand_unboxed, gold_unboxed, rel_tol=rel_tol)
            diagnostic = "Fraction equivalence match." if is_correct else f"Fraction mismatch: '{cand_str}' vs '{gold_str}'."

        elif eval_type == "float_tol":
            # Direct numeric float evaluation
            try:
                c_f = float(normalize_latex_expression(cand_unboxed))
                g_f = float(normalize_latex_expression(gold_unboxed))
                if math.isnan(c_f) or math.isnan(g_f):
                    is_correct = math.isnan(c_f) and math.isnan(g_f)
                elif math.isinf(c_f) or math.isinf(g_f):
                    is_correct = c_f == g_f
                else:
                    diff = abs(c_f - g_f)
                    is_correct = diff <= abs_tol or diff <= rel_tol * abs(g_f)
                diagnostic = "Float tolerance match." if is_correct else f"Float difference |{c_f} - {g_f}| exceeds tolerance."
            except ValueError:
                # Fallback to symbolic check
                is_correct = check_algebraic_equivalence(cand_unboxed, gold_unboxed, rel_tol=rel_tol, abs_tol=abs_tol)
                diagnostic = "Symbolic equivalence match." if is_correct else f"Evaluation mismatch: '{cand_str}' vs '{gold_str}'."

        else: # math_symbolic or exact fallback
            is_correct = check_algebraic_equivalence(
                cand_unboxed, gold_unboxed, rel_tol=rel_tol, abs_tol=abs_tol
            )
            diagnostic = "Symbolic mathematical equivalence match." if is_correct else f"Symbolic mismatch: '{cand_str}' vs '{gold_str}'."

    except ZeroDivisionError:
        is_correct = False
        diagnostic = "Division by zero encountered during mathematical evaluation."
    except Exception as e:
        is_correct = False
        diagnostic = f"Mathematical evaluation error: {type(e).__name__} ({str(e)})."

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return EvalResult(
        score=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        eval_type=eval_type,
        candidate_parsed=cand_unboxed,
        gold_target=gold_unboxed,
        diagnostic_message=diagnostic,
        execution_time_ms=round(elapsed_ms, 3),
        details={"rel_tol": rel_tol, "abs_tol": abs_tol, "raw_candidate": cand_str},
    )


class SympyMathEvaluator:
    """
    Object-oriented wrapper and unified interface for symbolic mathematical evaluation.
    Conforms to both the PROJECT.md interface contract and the E2E test suite contract.
    """

    def __init__(self, rel_tol: float = 0.01, abs_tol: float = 0.01):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    @staticmethod
    def evaluate(
        candidate: Any,
        ground_truth: Any,
        eval_type: str = "math_symbolic",
        rel_tol: float = 0.01,
        abs_tol: float = 0.01,
    ) -> float:
        """
        Evaluate mathematical equivalence returning a scalar score: 1.0 (correct) or 0.0 (incorrect).
        Conforms directly to E2E test suite contract.
        """
        result = evaluate_math_expression(
            candidate=candidate,
            gold=ground_truth,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            eval_type=eval_type,
        )
        return result.score

    @staticmethod
    def evaluate_detailed(
        candidate: Any,
        ground_truth: Any,
        eval_type: str = "math_symbolic",
        rel_tol: float = 0.01,
        abs_tol: float = 0.01,
    ) -> EvalResult:
        """Evaluate mathematical equivalence returning full EvalResult envelope."""
        return evaluate_math_expression(
            candidate=candidate,
            gold=ground_truth,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            eval_type=eval_type,
        )

    def evaluate_task(self, candidate: Any, ground_truth: Any, eval_type: str = "math_symbolic") -> EvalResult:
        """Instance method using instance default tolerances."""
        return evaluate_math_expression(
            candidate=candidate,
            gold=ground_truth,
            rel_tol=self.rel_tol,
            abs_tol=self.abs_tol,
            eval_type=eval_type,
        )
