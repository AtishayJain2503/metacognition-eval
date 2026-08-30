"""
tests.unit.test_eval.test_dataframe_diff
========================================
Unit tests for DataFrame diffing engine with shape checks, column alignment, and cell tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from nemo_eval.eval.dataframe_diff import (
    align_dataframe_columns,
    coerce_to_dataframe,
    compare_cell_values,
    evaluate_dataframe,
)


class TestDataFrameDiffCoercionAndAlignment:
    """Test coercing dicts/lists to DataFrames and column alignment."""

    def test_coerce_to_dataframe(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        df = coerce_to_dataframe(records)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_align_dataframe_columns_case_and_whitespace(self):
        df_cand = pd.DataFrame({" Product ": [1, 2], "Price": [10.0, 20.0]})
        df_gold = pd.DataFrame({"product": [1, 2], "price": [10.0, 20.0]})
        aligned_cand, aligned_gold, matched = align_dataframe_columns(df_cand, df_gold)
        assert matched is True
        assert list(aligned_cand.columns) == list(aligned_gold.columns)


class TestDataFrameDiffCellLevel:
    """Test comparing cell values and tolerance."""

    def test_compare_cell_values_numeric_tolerance(self):
        assert compare_cell_values(100.5, 100.0, rel_tol=0.01) is True
        assert compare_cell_values(105.0, 100.0, rel_tol=0.01) is False

    def test_compare_cell_values_nan_alignment(self):
        assert compare_cell_values(np.nan, np.nan) is True
        assert compare_cell_values(None, np.nan) is True
        assert compare_cell_values(10.0, np.nan) is False


class TestDataFrameDiffEngine:
    """Test full DataFrame evaluation diffs."""

    def test_exact_dataframe_match(self):
        df1 = pd.DataFrame({"id": [1, 2], "score": [95.0, 88.0]})
        df2 = pd.DataFrame({"id": [1, 2], "score": [95.0, 88.0]})
        res = evaluate_dataframe(df1, df2)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_dataframe_shape_mismatch(self):
        df1 = pd.DataFrame({"id": [1, 2, 3]})
        df2 = pd.DataFrame({"id": [1, 2]})
        res = evaluate_dataframe(df1, df2)
        assert res.is_correct is False
        assert "shape mismatch" in res.diagnostic_message

    def test_dataframe_cell_diff_with_tolerance(self):
        df1 = pd.DataFrame({"val": [10.05, 20.0]})
        df2 = pd.DataFrame({"val": [10.00, 20.0]})
        res = evaluate_dataframe(df1, df2, rel_tol=0.01)
        assert res.is_correct is True

        # Out of tolerance
        res_fail = evaluate_dataframe(df1, df2, rel_tol=0.001, abs_tol=0.001)
        assert res_fail.is_correct is False

    def test_dataframe_unordered_rows_match(self):
        df1 = pd.DataFrame({"name": ["Bob", "Alice"], "val": [20, 10]})
        df2 = pd.DataFrame({"name": ["Alice", "Bob"], "val": [10, 20]})
        res = evaluate_dataframe(df1, df2, check_order=False)
        assert res.is_correct is True
