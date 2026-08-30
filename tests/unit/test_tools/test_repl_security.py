"""
tests.unit.test_tools.test_repl_security
----------------------------------------
Unit tests for AST static security analysis and sandbox perimeter validation.
"""

import pytest

from nemo_eval.tools.repl import (
    CodeSecurityValidator,
    PythonREPL,
    SecurityViolationError,
)


class TestREPLSecurity:
    """Rigorous tests for AST security validation against sandbox escapes."""

    @pytest.fixture
    def validator(self):
        return CodeSecurityValidator()

    def test_forbidden_module_imports(self, validator):
        forbidden_snippets = [
            "import os\nos.system('whoami')",
            "import sys\nsys.exit(0)",
            "import subprocess\nsubprocess.run(['ls'])",
            "import socket\ns = socket.socket()",
            "import ctypes\nctypes.CDLL(None)",
            "import shutil\nshutil.rmtree('/tmp')",
            "import urllib.request\nurllib.request.urlopen('http://example.com')",
            "from os import path",
            "from subprocess import Popen",
            "from socket import gethostname",
        ]
        for snippet in forbidden_snippets:
            with pytest.raises(SecurityViolationError) as exc_info:
                validator.validate(snippet)
            assert "prohibited" in str(exc_info.value).lower()

    def test_forbidden_dunder_introspection(self, validator):
        dunder_snippets = [
            "subclasses = ().__class__.__bases__[0].__subclasses__()",
            "f = (lambda: None).__globals__",
            "c = (lambda: None).__code__",
            "b = ().__class__.__builtins__",
            "d = [].__dict__",
            "m = int.__mro__",
        ]
        for snippet in dunder_snippets:
            with pytest.raises(SecurityViolationError) as exc_info:
                validator.validate(snippet)
            assert "prohibited" in str(exc_info.value).lower()

    def test_forbidden_builtin_calls(self, validator):
        call_snippets = [
            "open('/etc/passwd', 'r')",
            "eval('1 + 2')",
            "exec('x = 5')",
            "compile('1+1', '<test>', 'eval')",
            "getattr(math, 'sin')",
            "setattr(obj, 'attr', 10)",
            "delattr(obj, 'attr')",
            "hasattr(obj, 'attr')",
            "breakpoint()",
            "input('enter name:')",
        ]
        for snippet in call_snippets:
            with pytest.raises(SecurityViolationError) as exc_info:
                validator.validate(snippet)
            assert "prohibited" in str(exc_info.value).lower()

    def test_safe_analytical_code_allowed(self, validator):
        safe_snippets = [
            "import math\nx = math.sqrt(25)",
            "import json\nd = json.loads('{\"a\": 1}')",
            "import re\nmatch = re.match(r'\\d+', '123')",
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
            "import numpy as np\narr = np.array([10, 20, 30])",
            "def calculate_stats(vals):\n    return sum(vals) / len(vals)\ncalculate_stats([1, 2, 3, 4])",
            "squares = [i**2 for i in range(10) if i % 2 == 0]",
        ]
        for snippet in safe_snippets:
            validator.validate(snippet)  # Should not raise

    def test_python_repl_security_envelope(self, repl_tool):
        res = repl_tool.execute("import os\nos.system('echo test')")
        assert res.status == "error"
        assert res.error is not None
        assert res.error.error_type == "SecurityViolation"
        assert "os" in res.error.message.lower() or "os" in res.error.suggestion.lower()
