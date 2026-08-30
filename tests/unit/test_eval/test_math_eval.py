"""
tests.unit.test_eval.test_math_eval
===================================
Comprehensive unit tests for SympyMathEvaluator and LaTeX mathematical equivalence engine.
"""

import pytest

from nemo_eval.eval.base import EvalResult
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


class TestLatexNormalization:
    """Test suite for LaTeX mathematical preprocessing and normalization."""

    def test_extract_latex_boxed(self):
        assert extract_latex_boxed(r"\boxed{42}") == "42"
        assert extract_latex_boxed(r"Therefore, \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert extract_latex_boxed(r"\boxed{\sqrt{x^2 + 1}}") == r"\sqrt{x^2 + 1}"
        assert extract_latex_boxed(r"First \boxed{10} and then \boxed{20}") == "20"
        assert extract_latex_boxed("No boxed expression here") is None

    def test_normalize_fractions(self):
        norm = normalize_latex_expression(r"\frac{3}{4}")
        assert "3" in norm and "4" in norm
        assert "/" in norm
        norm_dfrac = normalize_latex_expression(r"\dfrac{2x + 1}{3}")
        assert "(2x + 1)" in norm_dfrac or "2x + 1" in norm_dfrac

    def test_normalize_roots(self):
        assert "sqrt(50)" in normalize_latex_expression(r"\sqrt{50}")
        assert "sqrt(x**2 + 1)" in normalize_latex_expression(r"\sqrt{x^2 + 1}")
        norm_cube = normalize_latex_expression(r"\sqrt[3]{8}")
        assert "8" in norm_cube and "1/(3)" in norm_cube

    def test_normalize_constants_and_symbols(self):
        norm_pi = normalize_latex_expression(r"36\pi")
        assert "pi" in norm_pi
        norm_inf = normalize_latex_expression(r"\infty")
        assert "oo" in norm_inf
        norm_times = normalize_latex_expression(r"3 \times 4 \cdot 5")
        assert "*" in norm_times

    def test_normalize_trigonometric_powers(self):
        norm_sin2 = normalize_latex_expression(r"\sin^2(x)")
        assert "sin(x)" in norm_sin2 and "**2" in norm_sin2
        norm_cos2 = normalize_latex_expression(r"\cos^2(x)")
        assert "cos(x)" in norm_cos2 and "**2" in norm_cos2

    def test_normalize_formatting_and_markdown(self):
        assert normalize_latex_expression(r"\text{Answer: } 42") == "42"
        assert normalize_latex_expression(r"\mathbf{100}") == "100"
        assert normalize_latex_expression(r"**50**") == "50"
        assert normalize_latex_expression(r"`2*x + 1`") == "2*x + 1"


class TestAlgebraicEquivalence:
    """Test suite for algebraic and symbolic equivalence."""

    @pytest.mark.parametrize("cand,gold", [
        ("2*(x + 3)", "2*x + 6"),
        ("(x + 1)^2", "x^2 + 2*x + 1"),
        ("x^2 - 4", "(x - 2)*(x + 2)"),
        ("3*x + 2*x", "5*x"),
        ("4*(x - 1) + 2", "4*x - 2"),
    ])
    def test_polynomial_equivalence(self, cand, gold):
        assert check_algebraic_equivalence(cand, gold) is True
        res = evaluate_math_expression(cand, gold, eval_type="math_symbolic")
        assert res.is_correct is True
        assert res.score == 1.0

    @pytest.mark.parametrize("cand,gold", [
        (r"\frac{x^2 - 1}{x - 1}", "x + 1"),
        (r"\frac{2x + 4}{2}", "x + 2"),
        (r"\frac{6x^2}{2x}", "3*x"),
    ])
    def test_rational_expression_equivalence(self, cand, gold):
        assert check_algebraic_equivalence(cand, gold) is True
        res = evaluate_math_expression(cand, gold, eval_type="math_symbolic")
        assert res.is_correct is True

    def test_trigonometric_identities(self):
        assert check_algebraic_equivalence(r"\sin^2(x) + \cos^2(x)", "1") is True
        assert check_algebraic_equivalence(r"\sec^2(x) - \tan^2(x)", "1") is True
        assert check_algebraic_equivalence(r"\sin(2*x)", r"2*\sin(x)*\cos(x)") is True

    @pytest.mark.parametrize("cand,gold", [
        ("3/6", "1/2"),
        ("0.5", "1/2"),
        (r"\frac{1}{2}", "0.5"),
        ("2/4", "4/8"),
        ("0.75", "3/4"),
    ])
    def test_fraction_equivalence(self, cand, gold):
        assert check_fraction_equivalence(cand, gold) is True
        res = evaluate_math_expression(cand, gold, eval_type="fraction")
        assert res.is_correct is True

    @pytest.mark.parametrize("cand,gold,expected", [
        ("100.5", "100.0", True),    # 0.5% diff <= 1% rel_tol
        ("101.0", "100.0", True),    # 1.0% diff <= 1% rel_tol
        ("102.0", "100.0", False),   # 2.0% diff > 1% rel_tol
        ("0.005", "0.0", True),      # diff <= 0.01 abs_tol
    ])
    def test_numerical_tolerance_matching(self, cand, gold, expected):
        res = evaluate_math_expression(cand, gold, rel_tol=0.01, abs_tol=0.01, eval_type="float_tol")
        assert res.is_correct is expected

    @pytest.mark.parametrize("cand,gold", [
        (r"\{1, 2\}", r"\{2, 1\}"),
        (r"\{2, 3, 5\}", r"\{5, 2, 3\}"),
        (r"\{\sqrt{4}, 3\}", r"\{2, 3\}"),
        (r"\{\frac{1}{2}, 1\}", r"\{0.5, 1\}"),
    ])
    def test_set_equivalence(self, cand, gold):
        assert check_set_and_interval_equivalence(cand, gold) is True
        res = evaluate_math_expression(cand, gold, eval_type="set")
        assert res.is_correct is True

    def test_factorials_and_percentages(self):
        assert check_algebraic_equivalence("3!", "6") is True
        assert check_algebraic_equivalence("4!", "24") is True
        assert check_fraction_equivalence("25%", "0.25") is True
        assert check_fraction_equivalence(r"50\%", "1/2") is True


class TestRobustnessAndEdgeCases:
    """Test suite for boundary conditions, invalid inputs, and error recovery."""

    def test_syntax_error_unparseable_string(self):
        res = evaluate_math_expression("(((++**", "42")
        assert res.is_correct is False
        assert res.score == 0.0
        assert "mismatch" in res.diagnostic_message.lower() or "error" in res.diagnostic_message.lower()

    def test_division_by_zero_handled_safely(self):
        res = evaluate_math_expression("1/0", "42")
        assert res.is_correct is False
        assert res.score == 0.0

    def test_none_and_empty_inputs(self):
        res_none = evaluate_math_expression(None, "42")
        assert res_none.is_correct is False
        assert "None" in res_none.diagnostic_message

        res_empty = evaluate_math_expression("", "42")
        assert res_empty.is_correct is False
        assert "empty" in res_empty.diagnostic_message.lower()

    def test_complex_numbers_equivalence(self):
        cand = "3 + 4*I"
        gold = "3 + 4*I"
        res = evaluate_math_expression(cand, gold)
        assert res.is_correct is True
        assert res.score == 1.0


class TestSympyMathEvaluatorClass:
    """Test suite for SympyMathEvaluator class wrapper."""

    def test_evaluator_class_static_evaluate(self):
        score_correct = SympyMathEvaluator.evaluate(r"\boxed{2x + 4}", "2*(x + 2)")
        assert score_correct == 1.0

        score_incorrect = SympyMathEvaluator.evaluate("5", "10")
        assert score_incorrect == 0.0

    def test_evaluator_class_evaluate_detailed(self):
        res = SympyMathEvaluator.evaluate_detailed(r"\boxed{42}", "42")
        assert isinstance(res, EvalResult)
        assert res.is_correct is True
        assert res.score == 1.0
        assert res.execution_time_ms >= 0.0

    def test_evaluator_class_instance_method(self):
        evaluator = SympyMathEvaluator(rel_tol=0.05, abs_tol=0.05)
        res = evaluator.evaluate_task("104.0", "100.0", eval_type="float_tol")
        assert res.is_correct is True
