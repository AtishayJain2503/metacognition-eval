"""
tests.unit.test_tools.test_repl_timeout
---------------------------------------
Unit tests for process sandbox wall-clock timeouts and runaway execution termination.
"""

import time
import pytest

from nemo_eval.tools.repl import PythonREPL


class TestREPLTimeout:
    """Tests for hard wall-clock timeout handling in REPL sandbox."""

    def test_infinite_while_loop_timeout(self, repl_tool):
        start = time.perf_counter()
        res = repl_tool.execute("while True: pass", timeout=1.0)
        elapsed = time.perf_counter() - start

        assert elapsed < 3.0  # Finished within reasonable window of 1s timeout
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "TimeoutError"
        assert "timed out" in res.error.message.lower()

    def test_deep_recursion_or_busy_loop_timeout(self, repl_tool):
        code = """
def busy_fn(n):
    total = 0
    for i in range(100000000):
        total += (i % 7)
    return total
busy_fn(10)
"""
        res = repl_tool.execute(code, timeout=0.8)
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "TimeoutError"

    def test_recovery_after_timeout(self, repl_tool):
        # First execution times out
        res1 = repl_tool.execute("while True: pass", timeout=0.8)
        assert res1.status == "error"
        assert res1.error.error_type == "TimeoutError"

        # Subsequent execution in fresh session works without hangs
        res2 = repl_tool.execute("x = 50\nx * 2", timeout=3.0)
        assert res2.status == "success"
        assert res2.data == 100
