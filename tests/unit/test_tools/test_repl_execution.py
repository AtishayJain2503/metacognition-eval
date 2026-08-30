"""
tests.unit.test_tools.test_repl_execution
-----------------------------------------
Unit tests for REPL execution, dual-phase compilation, stream capture, and stateful sessions.
"""

import pytest

from nemo_eval.tools.repl import PythonREPL, compile_and_execute_ast


class TestREPLExecution:
    """Tests for dual-phase compilation, stdout capture, and stateful sessions."""

    def test_dual_phase_ast_compiler_direct(self):
        ns = {}
        # Case 1: Body statements + tail expression
        res1 = compile_and_execute_ast("a = 15\nb = 25\na + b", ns)
        assert ns.get("a") == 15
        assert ns.get("b") == 25
        assert res1 == 40

        # Case 2: Statement only
        res2 = compile_and_execute_ast("c = a * 2", ns)
        assert ns.get("c") == 30
        assert res2 is None

    def test_repl_terminal_expression_evaluation(self, repl_tool):
        res = repl_tool.execute("x = 10\ny = 20\nx * y")
        assert res.status == "success"
        assert res.data == 200

    def test_repl_stdout_and_stderr_capture(self, repl_tool):
        # Test standard stdout capture
        res = repl_tool.execute("print('Computed alpha value')\n42")
        assert res.status == "success"
        assert "Computed alpha value" in res.stdout
        assert res.data == 42

        # Test multiline stdout capture
        res_multi = repl_tool.execute("print('Line 1')\nprint('Line 2')\n'done'")
        assert res_multi.status == "success"
        assert "Line 1\nLine 2" in res_multi.stdout
        assert res_multi.data == "done"

    def test_stateful_session_persistence(self, repl_tool):
        sess_id = "session_test_persistence"
        # Turn 1: Define variables
        res1 = repl_tool.execute("base_value = 100\nmultiplier = 2.5", session_id=sess_id)
        assert res1.status == "success"

        # Turn 2: Use defined variables in subsequent turn
        res2 = repl_tool.execute("base_value * multiplier", session_id=sess_id)
        assert res2.status == "success"
        assert res2.data == 250.0

        # Turn 3: Define a function and call it
        res3 = repl_tool.execute("def compute_tax(amt):\n    return amt * 0.1\ncompute_tax(base_value)", session_id=sess_id)
        assert res3.status == "success"
        assert res3.data == 10.0

        repl_tool.close_session(sess_id)

    def test_session_isolation(self, repl_tool):
        sess_a = "session_A"
        sess_b = "session_B"

        repl_tool.execute("secret_var = 'alpha_secret'", session_id=sess_a)
        res_b = repl_tool.execute("secret_var", session_id=sess_b)
        
        assert res_b.status == "error"
        assert res_b.error is not None
        assert res_b.error.error_type == "NameError"

        repl_tool.close_session(sess_a)
        repl_tool.close_session(sess_b)

    def test_output_buffer_capping(self, repl_tool):
        # Generate 100,000 characters
        code = "print('X' * 100000)"
        res = repl_tool.execute(code)
        assert res.status == "success"
        assert len(res.stdout) < 60000
        assert "Output truncated" in res.stdout

    def test_pandas_dataframe_execution(self, repl_tool):
        code = """
import pandas as pd
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})
df.groupby('category')['sales'].sum().to_dict()
"""
        res = repl_tool.execute(code)
        assert res.status == "success"
        assert res.data == {"A": 250, "B": 450}
