"""
Unit tests for nemo_eval.telemetry.extractor (Milestone M1 - Value-Only Answer Extractor).
"""

import pytest
from nemo_eval.telemetry.extractor import ValueExtractor


class TestValueExtractor:
    """Test 7-tier value extraction and normalization."""

    def test_extract_boxed_simple(self):
        assert ValueExtractor.extract_value(r"The answer is \boxed{42}.") == "42"
        assert ValueExtractor.extract_value(r"\boxed{-15.5}") == "-15.5"

    def test_extract_boxed_nested_latex(self):
        raw = r"Therefore, the solution is \boxed{\frac{\sqrt{x^2+1}}{2}}."
        assert ValueExtractor.extract_value(raw) == r"\frac{\sqrt{x^2+1}}{2}"

    def test_extract_boxed_nested_curly_braces(self):
        raw = r"\boxed{\{1, 2, 3\}}"
        assert ValueExtractor.extract_value(raw) == r"\{1, 2, 3\}"

    def test_extract_boxed_deeply_nested(self):
        raw = r"\boxed{\frac{\sqrt{\frac{a}{b} + 1}}{c^2 + \frac{1}{d}}}"
        assert ValueExtractor.extract_value(raw) == r"\frac{\sqrt{\frac{a}{b} + 1}}{c^2 + \frac{1}{d}}"

    def test_extract_boxed_multiple_steps_picks_last(self):
        raw = r"First step gives \boxed{10}. Later step gives \boxed{42}."
        assert ValueExtractor.extract_value(raw) == "42"

    def test_extract_json_markdown_code_block(self):
        raw = "```json\n{\n  \"answer\": \"100\"\n}\n```"
        assert ValueExtractor.extract_value(raw) == "100"

    def test_extract_json_inline_final_answer(self):
        raw = 'Execution completed. {"final_answer": "3.14159"}'
        assert ValueExtractor.extract_value(raw) == "3.14159"

    def test_extract_anchor_final_answer(self):
        assert ValueExtractor.extract_value("Final Answer: 3.14159") == "3.14159"
        assert ValueExtractor.extract_value("final answer is: 200") == "200"

    def test_extract_anchor_the_answer_is(self):
        assert ValueExtractor.extract_value("Therefore, the answer is 99.") == "99"
        assert ValueExtractor.extract_value("Thus, the answer is 50") == "50"

    def test_extract_anchor_hash_delimiter(self):
        assert ValueExtractor.extract_value("Step 1\nStep 2\n#### 12345") == "12345"

    def test_extract_equation_tail(self):
        assert ValueExtractor.extract_value("After algebraic evaluation, x = 105") == "105"
        assert ValueExtractor.extract_value("We get ans = 2*pi") == "2*pi"

    def test_extract_markdown_and_prose_stripping(self):
        assert ValueExtractor.extract_value("**\\boxed{42}**") == "42"
        assert ValueExtractor.extract_value("The answer is **$99**.") == "99"
        assert ValueExtractor.extract_value("`42`") == "42"

    def test_extract_currency_and_thousands_commas(self):
        assert ValueExtractor.extract_value("$1,250,000.50") == "1250000.50"
        assert ValueExtractor.extract_value("Cost: €4,500") == "4500"

    def test_extract_prose_with_units(self):
        assert ValueExtractor.extract_value("The distance is 100 meters.") == "100"
        assert ValueExtractor.extract_value("Efficiency is 85%") == "85"
        assert ValueExtractor.extract_value("Energy: 50 Joules.") == "50"
        assert ValueExtractor.extract_value("Count: 15 animals.") == "15"

    def test_extract_standalone_numeric_fallback(self):
        assert ValueExtractor.extract_value("We counted 42 sheep in the meadow.") == "42"

    def test_extract_fractions_and_scientific(self):
        assert ValueExtractor.extract_value("Result: 3/4") == "3/4"
        assert ValueExtractor.extract_value("Value is 1.5e-4") == "1.5e-4"

    def test_extract_empty_and_none(self):
        assert ValueExtractor.extract_value("") == ""
        assert ValueExtractor.extract_value(None) == ""
        assert ValueExtractor.extract_value("   \n\t  ") == ""

    def test_extract_malformed_unbalanced_boxed(self):
        assert ValueExtractor.extract_value(r"\boxed{42") == "42"

    def test_expected_type_set_conditioning(self):
        assert ValueExtractor.extract_value("1, 2, 3", expected_type="set") == "{1, 2, 3}"
        assert ValueExtractor.extract_value("{1, 2, 3}", expected_type="set") == "{1, 2, 3}"

    def test_think_tag_isolation(self):
        raw = "<think>\nLet's solve step by step: 2+2=4.\n</think>\nThe answer is \\boxed{4}"
        assert ValueExtractor.extract_value(raw) == "4"
