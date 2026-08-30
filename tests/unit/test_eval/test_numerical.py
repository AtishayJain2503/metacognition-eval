"""
tests.unit.test_eval.test_numerical
===================================
Unit tests for dual relative/absolute tolerance, zero reference, NaN, inf, and percentage scaling.
"""

import math
import pytest

from nemo_eval.eval.numerical import (
    check_numerical_tolerance,
    evaluate_numerical,
    extract_numerical_value,
)


class TestNumericalExtraction:
    """Test extracting clean float values from various string encodings and units."""

    def test_extract_floats_and_ints(self):
        assert extract_numerical_value(42) == 42.0
        assert extract_numerical_value(3.14159) == 3.14159
        assert extract_numerical_value("100.5") == 100.5
        assert extract_numerical_value("-15.2") == -15.2

    def test_extract_units_and_currencies(self):
        assert extract_numerical_value("$1,250.00") == 1250.0
        assert extract_numerical_value("45.6 kg") == 45.6
        assert extract_numerical_value("98.5%") == 98.5
        assert extract_numerical_value("€ 450,000") == 450000.0

    def test_extract_scientific_and_latex(self):
        assert extract_numerical_value("1.25e-4") == 0.000125
        assert extract_numerical_value("\\boxed{350.75}") == 350.75
        assert extract_numerical_value("\\text{Score: 88.0}") == 88.0

    def test_extract_special_values(self):
        assert math.isinf(extract_numerical_value("inf"))
        assert math.isinf(extract_numerical_value("-infinity"))
        assert math.isnan(extract_numerical_value("nan"))


class TestNumericalDualTolerance:
    """Test relative tolerance (eps=0.01) and absolute tolerance (delta=0.01)."""

    def test_exact_float_equality(self):
        res = evaluate_numerical(100.0, 100.0)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_relative_tolerance_boundary(self):
        # eps = 0.01 (1%) -> on 100.0, 99.0 to 101.0 is valid
        res_pass = evaluate_numerical(100.9, 100.0, rel_tol=0.01, abs_tol=0.001)
        assert res_pass.is_correct is True

        res_fail = evaluate_numerical(101.5, 100.0, rel_tol=0.01, abs_tol=0.001)
        assert res_fail.is_correct is False

    def test_absolute_tolerance_boundary_near_zero(self):
        # When gold is 0.0, relative tolerance fails and absolute tolerance governs
        res_pass = evaluate_numerical(0.008, 0.0, abs_tol=0.01)
        assert res_pass.is_correct is True

        res_fail = evaluate_numerical(0.05, 0.0, abs_tol=0.01)
        assert res_fail.is_correct is False

    def test_negative_values(self):
        res = evaluate_numerical(-50.4, -50.0, rel_tol=0.01)
        assert res.is_correct is True


class TestNumericalSpecialEdgeCases:
    """Test NaN, Infinity, signed zero, and percentage conversion."""

    def test_nan_handling(self):
        res_both_nan = evaluate_numerical(float("nan"), float("nan"))
        assert res_both_nan.is_correct is True

        res_one_nan = evaluate_numerical(float("nan"), 0.0)
        assert res_one_nan.is_correct is False

    def test_infinity_handling(self):
        res_pos_inf = evaluate_numerical(float("inf"), float("inf"))
        assert res_pos_inf.is_correct is True

        res_diff_inf = evaluate_numerical(float("inf"), float("-inf"))
        assert res_diff_inf.is_correct is False

    def test_signed_zero(self):
        res = evaluate_numerical(-0.0, 0.0)
        assert res.is_correct is True

    def test_percentage_scaling(self):
        # 0.235 vs 23.5%
        res1 = evaluate_numerical("23.5%", 0.235)
        assert res1.is_correct is True

        # 25 vs 0.25
        res2 = evaluate_numerical(25, 0.25)
        assert res2.is_correct is True

        # 0.5 vs 50.0
        res3 = evaluate_numerical(0.50, 50.0)
        assert res3.is_correct is True
