"""
tests.unit.test_tools.test_diagnostics
--------------------------------------
Unit tests for error classifier, visual syntax caret pointers, and remediation suggestions.
"""

import sqlite3
import pytest

from nemo_eval.tools.diagnostics import DiagnosticClassifier, DiagnosticFormatter
from nemo_eval.tools.repl import SecurityViolationError


class TestDiagnosticClassifier:
    """Tests for exception classification, pointer highlighting, and fuzzy remediation."""

    def test_classify_standard_exceptions(self):
        assert DiagnosticClassifier.classify_error_type(SyntaxError("bad syntax")) == "SyntaxError"
        assert DiagnosticClassifier.classify_error_type(NameError("name 'x' is not defined")) == "NameError"
        assert DiagnosticClassifier.classify_error_type(KeyError("missing_key")) == "KeyError"
        assert DiagnosticClassifier.classify_error_type(IndexError("list index out of range")) == "IndexError"
        assert DiagnosticClassifier.classify_error_type(ZeroDivisionError("division by zero")) == "ZeroDivisionError"
        assert DiagnosticClassifier.classify_error_type(TimeoutError("timed out")) == "TimeoutError"
        assert DiagnosticClassifier.classify_error_type(sqlite3.OperationalError("no such table")) == "OperationalError"
        assert DiagnosticClassifier.classify_error_type(SecurityViolationError("forbidden")) == "SecurityViolation"

    def test_python_syntax_error_caret_pointer(self):
        code = "def foo():\n    x = 10 +\n    return x"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as e:
            lineno, col, line_text, pointer = DiagnosticClassifier.highlight_python_syntax_error(e, code)
            assert lineno == 2
            assert line_text == "    x = 10 +"
            assert "^" in pointer

    def test_sql_syntax_error_caret_pointer(self):
        sql = "SELECT id, name\nFROM custmers\nWHERE id = 1;"
        err_msg = "no such table: custmers"
        lineno, col, line_text, pointer = DiagnosticClassifier.highlight_sql_error(sql, err_msg)
        assert lineno == 2
        assert "FROM custmers" in line_text
        assert "^^^^^^^^" in pointer

    def test_name_error_fuzzy_remediation(self):
        # Case 1: Builtin close match (sqt -> sqrt)
        hint = DiagnosticClassifier.suggest_name_remediation("sqt", session_vars=["df_sales", "total_sum"])
        assert "sqrt" in hint
        assert "Did you mean" in hint

        # Case 2: Session var close match (df_sale -> df_sales)
        hint2 = DiagnosticClassifier.suggest_name_remediation("df_sale", session_vars=["df_sales", "customers"])
        assert "df_sales" in hint2

    def test_key_error_fuzzy_remediation(self):
        columns = ["customer_id", "first_name", "last_name", "total_spend"]
        hint = DiagnosticClassifier.suggest_key_remediation("cust_id", available_keys=columns)
        assert "customer_id" in hint
        assert "Did you mean" in hint

    def test_sql_table_fuzzy_remediation(self):
        tables = ["customers", "orders", "order_items", "products"]
        hint = DiagnosticClassifier.suggest_sql_table_remediation("custmers", available_tables=tables)
        assert "customers" in hint
        assert "Did you mean 'customers'?" in hint

    def test_sql_column_fuzzy_remediation(self):
        cols = ["product_id", "product_name", "unit_price", "quantity"]
        hint = DiagnosticClassifier.suggest_sql_column_remediation("unit_prie", available_columns=cols, table_name="products")
        assert "unit_price" in hint
        assert "in table 'products'" in hint

    def test_traceback_sanitization(self):
        try:
            raise ValueError("Test error for sanitization")
        except ValueError as e:
            cleaned_tb = DiagnosticClassifier.sanitize_traceback(e, limit_frames=3)
            assert "ValueError: Test error for sanitization" in cleaned_tb
            assert "_pytest" not in cleaned_tb

    def test_create_diagnostic_error_full_flow(self):
        code = "a = 10\nb = sqt(16)"
        try:
            exec(code, {"a": 10})
        except NameError as exc:
            diag = DiagnosticClassifier.create_diagnostic_error(
                exc=exc,
                source_code=code,
                context={"session_vars": ["a", "b", "sqrt"]}
            )
            assert diag.error_type == "NameError"
            assert "sqrt" in diag.suggestion
            assert diag.line_number == 2
            assert "b = sqt(16)" in diag.code_snippet
            assert "^^^" in diag.pointer

    def test_readonly_database_diagnostic(self):
        exc = sqlite3.OperationalError("attempt to write a readonly database")
        diag = DiagnosticClassifier.create_diagnostic_error(
            exc=exc,
            source_code="INSERT INTO users VALUES (1, 'Alice');"
        )
        assert diag.error_type == "OperationalError"
        assert "read-only" in diag.suggestion.lower() or "readonly" in diag.suggestion.lower()
