"""
tests.unit.test_eval.test_exact
===============================
Unit tests for exact and normalized string, boolean, and collection matching.
"""

import pytest

from nemo_eval.eval.exact import (
    evaluate_exact,
    normalize_boolean,
    normalize_text,
    strip_articles,
    strip_markdown,
)


class TestExactStringMatching:
    """Test text normalizations, punctuation/markdown/currency stripping."""

    def test_strip_markdown(self):
        assert strip_markdown("**California**") == "California"
        assert strip_markdown("*apple*") == "apple"
        assert strip_markdown("`select * from table`") == "select * from table"
        assert strip_markdown("\\boxed{42.0}") == "42.0"
        assert strip_markdown("\\text{result}") == "result"

    def test_strip_articles(self):
        assert strip_articles("the United States") == "United States"
        assert strip_articles("a red car") == "red car"
        assert strip_articles("an orange") == "orange"

    def test_normalize_text_comprehensive(self):
        assert normalize_text("  $1,250,000.50  ") == "1250000.50"
        assert normalize_text("**The** Total is: 95.5%!!") == "total is 95.5"
        assert normalize_text("San Francisco, CA.") == "san francisco ca"

    def test_evaluate_exact_strings(self):
        # Exact match
        res1 = evaluate_exact("New York", "new york")
        assert res1.is_correct is True
        assert res1.score == 1.0

        # Punctuation and whitespace mismatch
        res2 = evaluate_exact("  **Amazon** Inc.  ", "amazon inc")
        assert res2.is_correct is True
        assert res2.score == 1.0

        # Truly different strings
        res3 = evaluate_exact("Google", "Microsoft")
        assert res3.is_correct is False
        assert res3.score == 0.0


class TestBooleanMatching:
    """Test boolean normalization and truthy/falsy equivalence."""

    def test_normalize_boolean(self):
        # Truthy
        assert normalize_boolean(True) is True
        assert normalize_boolean("True") is True
        assert normalize_boolean("YES") is True
        assert normalize_boolean("1") is True
        assert normalize_boolean(1) is True
        assert normalize_boolean("correct") is True

        # Falsy
        assert normalize_boolean(False) is False
        assert normalize_boolean("False") is False
        assert normalize_boolean("no") is False
        assert normalize_boolean("0") is False
        assert normalize_boolean(0) is False
        assert normalize_boolean("incorrect") is False

        # Non-boolean
        assert normalize_boolean("maybe") is None
        assert normalize_boolean(42) is None

    def test_evaluate_exact_booleans(self):
        res1 = evaluate_exact("yes", True)
        assert res1.is_correct is True
        assert res1.score == 1.0

        res2 = evaluate_exact("False", "no")
        assert res2.is_correct is True
        assert res2.score == 1.0

        res3 = evaluate_exact("True", False)
        assert res3.is_correct is False
        assert res3.score == 0.0


class TestCollectionMatching:
    """Test list, set, and tuple exact and unordered matching."""

    def test_unordered_set_matching(self):
        cand = ["Apples", "Bananas", "Oranges"]
        gold = ["oranges", "apples", "bananas"]
        res = evaluate_exact(cand, gold, unordered_collection=True)
        assert res.is_correct is True
        assert res.score == 1.0

    def test_ordered_list_matching(self):
        cand = ["A", "B", "C"]
        gold = ["A", "B", "C"]
        res = evaluate_exact(cand, gold, unordered_collection=False)
        assert res.is_correct is True

        # Mismatched order with strict ordering
        res_fail = evaluate_exact(["B", "A", "C"], ["A", "B", "C"], unordered_collection=False)
        assert res_fail.is_correct is False

    def test_collection_from_string_representation(self):
        cand = "['Alpha', 'Beta', 'Gamma']"
        gold = ["alpha", "beta", "gamma"]
        res = evaluate_exact(cand, gold)
        assert res.is_correct is True

    def test_collection_mismatch_counts(self):
        cand = ["A", "B"]
        gold = ["A", "B", "C"]
        res = evaluate_exact(cand, gold)
        assert res.is_correct is False
        assert res.score == 0.0
