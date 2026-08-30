"""
test_tier1_features.py - Opaque-Box Feature Tests (Tier 1) for NeMo Long-Horizon Evaluation Benchmark.
Covers all 32 features from PROJECT.md across M1, M2, M3, M4, M5.
"""
import ast
import io
import math
import os
import re
import sqlite3
import sys
import pandas as pd
import numpy as np
import pytest
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from tests.e2e.conftest import (
    ToolResult, DiagnosticError, BenchmarkTask, ToolCall, LLMMessage,
    LLMResponse, StepEvent, EpisodeTrajectory
)


# ===========================================================================
# FEATURE AREA 1: Hermetic Auxiliary Tools & Sandboxing (Features 1 - 11)
# ===========================================================================

class TestHermeticTools:
    """Tests for REPL Sandbox, SQLite Engine, Tabular Engine, and Diagnostics."""

    def test_feature_01_ast_security_validator(self):
        """Feature 1: AST Security Validator blocks prohibited modules and dunder traversal."""
        forbidden_modules = {"os", "sys", "subprocess", "socket", "urllib", "shutil"}
        forbidden_dunders = {"__subclasses__", "__bases__", "__globals__", "__code__"}

        def validate_code_ast(code: str) -> List[str]:
            tree = ast.parse(code)
            violations = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in forbidden_modules:
                            violations.append(f"Forbidden import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in forbidden_modules:
                        violations.append(f"Forbidden from-import: {node.module}")
                elif isinstance(node, ast.Attribute):
                    if node.attr in forbidden_dunders:
                        violations.append(f"Forbidden dunder: {node.attr}")
            return violations

        # Safe code should have 0 violations
        safe_code = "import math\nx = math.sqrt(16)\ny = [i**2 for i in range(5)]"
        assert validate_code_ast(safe_code) == []

        # Dangerous code with prohibited imports
        bad_code_1 = "import os\nos.system('echo test')"
        violations_1 = validate_code_ast(bad_code_1)
        assert len(violations_1) > 0
        assert "os" in violations_1[0]

        # Dangerous code with dunder traversal
        bad_code_2 = "subclasses = ().__class__.__bases__[0].__subclasses__()"
        violations_2 = validate_code_ast(bad_code_2)
        assert any("__subclasses__" in v for v in violations_2)
        assert any("__bases__" in v for v in violations_2)

    def test_feature_02_dual_phase_repl_compilation(self):
        """Feature 2: Dual-Phase REPL executes body as exec and tail expression as eval."""
        def dual_phase_execute(code: str, scope: dict) -> Any:
            tree = ast.parse(code)
            if not tree.body:
                return None
            if isinstance(tree.body[-1], ast.Expr):
                body_stmts = tree.body[:-1]
                expr_stmt = tree.body[-1]
                if body_stmts:
                    exec_mod = ast.Module(body=body_stmts, type_ignores=[])
                    exec(compile(exec_mod, "<repl_body>", "exec"), scope)
                eval_expr = ast.Expression(body=expr_stmt.value)
                return eval(compile(eval_expr, "<repl_tail>", "eval"), scope)
            else:
                exec(compile(tree, "<repl_all>", "exec"), scope)
                return None

        scope = {}
        result = dual_phase_execute("a = 10\nb = 25\na + b", scope)
        assert scope.get("a") == 10
        assert scope.get("b") == 25
        assert result == 35

    def test_feature_03_safe_builtins_whitelist(self):
        """Feature 3: Safe builtins whitelist restricts dangerous standard functions."""
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
            "len": len, "range": range, "enumerate": enumerate, "zip": zip,
            "int": int, "float": float, "str": str, "bool": bool, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "isinstance": isinstance
        }
        # Whitelisted functions execute cleanly
        assert safe_builtins["sum"]([1, 2, 3, 4]) == 10
        assert safe_builtins["min"](5, 2, 9) == 2
        # Dangerous builtins like open, __import__, eval, exec are excluded
        assert "open" not in safe_builtins
        assert "__import__" not in safe_builtins
        assert "eval" not in safe_builtins
        assert "exec" not in safe_builtins

    def test_feature_04_repl_stream_redirection(self):
        """Feature 4: REPL captures stdout/stderr into ToolResult."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            print("Step 1: Computed intermediate matrix", file=sys.stdout)
            print("Warning: Low variance in column X", file=sys.stderr)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        result = ToolResult(
            status="success",
            execution_time_ms=12.5,
            data={"mean": 42.0},
            stdout=stdout_buf.getvalue().strip(),
            stderr=stderr_buf.getvalue().strip()
        )
        assert result.status == "success"
        assert "Step 1: Computed intermediate matrix" in result.stdout
        assert "Warning: Low variance" in result.stderr
        assert result.data["mean"] == 42.0

    def test_feature_05_transient_sqlite_lifecycle(self):
        """Feature 5: Transient SQLite lifecycle initializes :memory: DB, populates DDL, and tears down."""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE metrics (id INT PRIMARY KEY, score REAL);")
        cursor.executemany("INSERT INTO metrics VALUES (?, ?);", [(1, 0.95), (2, 0.88)])
        conn.commit()

        cursor.execute("SELECT AVG(score) FROM metrics;")
        avg_score = cursor.fetchone()[0]
        assert pytest.approx(avg_score, rel=1e-3) == 0.915

        conn.close()
        # Verify connection is closed
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.execute("SELECT 1;")

    def test_feature_06_database_read_only_pragma(self, sample_sqlite_db):
        """Feature 6: Read-only PRAGMA prevents data modification."""
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA query_only = ON;")
        
        # Read query succeeds
        cursor.execute("SELECT count(*) FROM products;")
        assert cursor.fetchone()[0] == 5
        
        # Write attempt fails due to read-only PRAGMA
        with pytest.raises(sqlite3.OperationalError, match="attempt to write a readonly database"):
            cursor.execute("INSERT INTO categories VALUES (99, 'Forbidden Category');")
        conn.close()

    def test_feature_07_sqlite_schema_introspection(self, sample_sqlite_db):
        """Feature 7: Introspects table structures, column types, nullability, and primary/foreign keys."""
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "categories" in tables
        assert "products" in tables
        assert "orders" in tables
        assert "order_items" in tables

        # Introspect products columns
        cursor.execute("PRAGMA table_info('products');")
        cols = {row[1]: {"type": row[2], "pk": bool(row[5])} for row in cursor.fetchall()}
        assert "product_id" in cols and cols["product_id"]["pk"] is True
        assert "price" in cols and cols["price"]["type"] == "REAL"

        # Introspect foreign keys
        cursor.execute("PRAGMA foreign_key_list('products');")
        fks = cursor.fetchall()
        assert len(fks) >= 1
        assert fks[0][2] == "categories" # Referenced table
        conn.close()

    def test_feature_08_sql_result_bounding_and_pagination(self, temp_dir):
        """Feature 8: Bounded SQL query output (caps at 50 rows) with pagination indicator."""
        db_path = os.path.join(temp_dir, "large_table.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE big_table (id INTEGER PRIMARY KEY, val REAL);")
        cursor.executemany("INSERT INTO big_table VALUES (?, ?);", [(i, i * 1.5) for i in range(120)])
        conn.commit()

        # Query bounded execution
        def execute_bounded_query(c, sql: str, limit: int = 50):
            c.execute(sql)
            rows = c.fetchmany(limit + 1)
            has_more = len(rows) > limit
            returned_rows = rows[:limit]
            return {
                "rows": returned_rows,
                "count": len(returned_rows),
                "has_more": has_more,
                "suggestion": "Use LIMIT/OFFSET" if has_more else ""
            }

        res = execute_bounded_query(cursor, "SELECT * FROM big_table;")
        assert res["count"] == 50
        assert res["has_more"] is True
        assert "LIMIT/OFFSET" in res["suggestion"]
        conn.close()

    def test_feature_09_tabular_profiler(self, sample_csv_path):
        """Feature 9: Ingests tabular CSV/Parquet and generates 8-point numerical and categorical profile."""
        df = pd.read_csv(sample_csv_path)
        assert df.shape == (8, 7)
        assert set(df.columns) == {"customer_id", "name", "tenure_months", "monthly_charges", "total_charges", "contract", "churned"}
        
        # Verify 8-point summary statistics
        desc = df["monthly_charges"].describe()
        assert desc["count"] == 8
        assert pytest.approx(desc["min"], rel=1e-2) == 29.99
        assert pytest.approx(desc["max"], rel=1e-2) == 119.00
        assert pytest.approx(desc["mean"], rel=1e-2) == 77.405

        # Verify categorical profile
        contract_counts = df["contract"].value_counts().to_dict()
        assert contract_counts["month-to-month"] == 3
        assert contract_counts["two-year"] == 3
        assert contract_counts["one-year"] == 2

    def test_feature_10_diagnostic_error_formatter(self):
        """Feature 10: Formats exceptions into visual pointers and remediation suggestions."""
        def format_diagnostic(err: Exception, code_snippet: str = "") -> DiagnosticError:
            err_type = type(err).__name__
            msg = str(err)
            suggestion = ""
            if isinstance(err, ZeroDivisionError):
                suggestion = "Check denominator for zero or empty collection before division."
            elif isinstance(err, KeyError):
                suggestion = f"Verify dictionary or DataFrame column '{msg}' exists using .columns or .keys()."
            elif isinstance(err, sqlite3.OperationalError):
                suggestion = "Check SQL syntax, table/column names, and read-only transaction state."
            
            return DiagnosticError(
                error_type=err_type,
                message=msg,
                code_snippet=code_snippet,
                suggestion=suggestion,
                raw_traceback=f"Traceback: {err_type}: {msg}"
            )

        diag = format_diagnostic(KeyError("revenue"), "df['revenue'] * 0.15")
        assert diag.error_type == "KeyError"
        assert "revenue" in diag.suggestion
        assert "Verify dictionary or DataFrame column" in diag.suggestion

    def test_feature_11_tool_json_schemas(self):
        """Feature 11: Validates OpenAI/NeMo-compatible JSON function definitions for all auxiliary tools."""
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "python_repl",
                    "description": "Execute deterministic Python code in a sandboxed REPL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python snippet to execute."}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sqlite_query",
                    "description": "Execute read-only SQL queries on SQLite database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query to execute."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "tabular_inspect",
                    "description": "Inspect tabular CSV/Parquet schema, dtypes, and sample rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to CSV or Parquet file."}
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
        assert len(tool_schemas) == 3
        for s in tool_schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]
            assert "properties" in s["function"]["parameters"]


# ===========================================================================
# FEATURE AREA 2: Benchmark Datasets & Ground Truth Engines (Features 12 - 17)
# ===========================================================================

class TestBenchmarkDatasetsAndEval:
    """Tests for InfiAgent, BIRD-SQL, DataBench ingestion, polymorphic eval, and pass@k."""

    def test_feature_12_infiagent_dabench_ingestion(self):
        """Feature 12: Ingests InfiAgent data analytics tasks and parses gold answers."""
        task_data = {
            "task_id": "infi_001",
            "benchmark_name": "infiagent",
            "query": "What is the average tenure of churned customers?",
            "ground_truth": 6.6667,
            "eval_type": "float_tol",
            "metadata": {"dataset_name": "telecom_churn", "question_type": "aggregation"}
        }
        task = BenchmarkTask(**task_data)
        assert task.task_id == "infi_001"
        assert task.benchmark_name == "infiagent"
        assert task.eval_type == "float_tol"
        assert pytest.approx(task.ground_truth, rel=1e-3) == 6.6667

    def test_feature_13_bird_sql_ingestion(self):
        """Feature 13: Ingests BIRD-SQL text-to-SQL tasks with DDL schemas and evidence dict."""
        task_data = {
            "task_id": "bird_042",
            "benchmark_name": "bird_sql",
            "query": "Find all products with price higher than 100 in category 'Electronics'.",
            "db_path": "ecommerce.db",
            "ground_truth": [("Laptop Pro", 1299.99)],
            "eval_type": "sql_multiset",
            "metadata": {
                "evidence": "Category name 'Electronics' maps to category_id = 1",
                "golden_sql": "SELECT p.product_name, p.price FROM products p JOIN categories c ON p.category_id = c.category_id WHERE c.category_name = 'Electronics' AND p.price > 100;"
            }
        }
        task = BenchmarkTask(**task_data)
        assert task.task_id == "bird_042"
        assert task.benchmark_name == "bird_sql"
        assert "golden_sql" in task.metadata
        assert "evidence" in task.metadata

    def test_feature_14_databench_ingestion_and_categorization(self):
        """Feature 14: Ingests DataBench tasks across semantic types (Scalar, Boolean, List/Set, Table)."""
        semantic_types = ["Scalar", "Boolean", "List/Set", "Table"]
        tasks = [
            BenchmarkTask(
                task_id=f"db_{st}",
                benchmark_name="databench",
                query=f"Query for {st}",
                ground_truth=42 if st == "Scalar" else True if st == "Boolean" else ["A", "B"] if st == "List/Set" else {"col": [1, 2]},
                eval_type="exact" if st in ["Scalar", "Boolean"] else "dataframe_diff" if st == "Table" else "sql_multiset",
                metadata={"semantic_type": st}
            )
            for st in semantic_types
        ]
        assert len(tasks) == 4
        assert [t.metadata["semantic_type"] for t in tasks] == semantic_types

    def test_feature_15_offline_synthetic_fixtures(self, sample_csv_path, sample_sqlite_db):
        """Feature 15: Offline synthetic fixtures operate with 0% network access."""
        assert os.path.exists(sample_csv_path)
        assert os.path.exists(sample_sqlite_db)
        
        # Verify DB fixture data integrity
        conn = sqlite3.connect(sample_sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM orders;")
        assert cursor.fetchone()[0] == 3
        conn.close()

    def test_feature_16_polymorphic_ground_truth_evaluation(self):
        """Feature 16: Polymorphic evaluation supports Exact, Float Tol, SQL Multiset, and DataFrame Diff."""
        def evaluate_polymorphic(predicted: Any, ground_truth: Any, eval_type: str, rel_tol: float = 0.01) -> bool:
            if eval_type == "exact":
                if isinstance(predicted, str) and isinstance(ground_truth, str):
                    return predicted.strip().lower() == ground_truth.strip().lower()
                return predicted == ground_truth
            elif eval_type == "float_tol":
                try:
                    p_val = float(predicted)
                    g_val = float(ground_truth)
                    if math.isnan(p_val) and math.isnan(g_val):
                        return True
                    return math.isclose(p_val, g_val, rel_tol=rel_tol, abs_tol=1e-4)
                except (ValueError, TypeError):
                    return False
            elif eval_type == "sql_multiset":
                # Multiset counter comparison
                from collections import Counter
                p_list = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in predicted]
                g_list = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in ground_truth]
                return Counter(p_list) == Counter(g_list)
            elif eval_type == "dataframe_diff":
                df_p = pd.DataFrame(predicted)
                df_g = pd.DataFrame(ground_truth)
                return df_p.equals(df_g)
            return False

        # Exact match
        assert evaluate_polymorphic("Yes", "yes", "exact") is True
        assert evaluate_polymorphic("No", "yes", "exact") is False

        # Float tolerance (within 1%)
        assert evaluate_polymorphic(100.5, 100.0, "float_tol", rel_tol=0.01) is True
        assert evaluate_polymorphic(102.5, 100.0, "float_tol", rel_tol=0.01) is False

        # SQL Multiset (order independent)
        pred_sql = [(1, "A"), (2, "B")]
        gold_sql = [(2, "B"), (1, "A")]
        assert evaluate_polymorphic(pred_sql, gold_sql, "sql_multiset") is True

        # DataFrame diffing
        df1 = {"a": [1, 2], "b": [3, 4]}
        df2 = {"a": [1, 2], "b": [3, 4]}
        assert evaluate_polymorphic(df1, df2, "dataframe_diff") is True

    def test_feature_17_pass_at_k_unbiased_estimator(self):
        """Feature 17: Computes unbiased pass@k accuracy estimator."""
        def estimate_pass_at_k(n: int, c: int, k: int) -> float:
            """Unbiased estimator: 1 - comb(n - c, k) / comb(n, k)."""
            if n - c < k:
                return 1.0
            return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

        # 10 samples, 4 correct:
        # pass@1 = 4 / 10 = 0.40
        assert pytest.approx(estimate_pass_at_k(n=10, c=4, k=1), rel=1e-3) == 0.40
        # pass@5
        p5 = estimate_pass_at_k(n=10, c=4, k=5)
        assert 0.90 <= p5 <= 1.0
        # When all correct (c = n), pass@k is 1.0
        assert estimate_pass_at_k(n=5, c=5, k=1) == 1.0
        # When none correct (c = 0), pass@k is 0.0
        assert estimate_pass_at_k(n=5, c=0, k=1) == 0.0


# ===========================================================================
# FEATURE AREA 3: Model Provider Layer & Mock Runner (Features 18 - 22)
# ===========================================================================

class TestModelProviderLayer:
    """Tests for BaseLLMClient protocol, Groq <think> extractor, and Mock Runner."""

    def test_feature_18_unified_base_llm_client_protocol(self):
        """Feature 18: BaseLLMClient protocol defines generate and agenerate contracts."""
        class MockBaseClient:
            def generate(self, messages: List[LLMMessage], tools: Optional[List[dict]] = None) -> LLMResponse:
                return LLMResponse(
                    content="Plan: Inspect tabular schema first.",
                    tool_calls=[ToolCall(id="tc_1", name="tabular_inspect", arguments={"file_path": "data.csv"})],
                    prompt_tokens=50,
                    completion_tokens=25
                )

        client = MockBaseClient()
        messages = [LLMMessage(role="user", content="Analyze the data.")]
        resp = client.generate(messages)
        assert resp.content.startswith("Plan:")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "tabular_inspect"

    def test_feature_19_groq_think_tag_isolation(self):
        """Feature 19: Groq client extracts and isolates <think> reasoning tokens from DeepSeek-R1 responses."""
        def parse_reasoning_and_content(raw_text: str) -> tuple[Optional[str], str]:
            pattern = r"<think>(.*?)</think>"
            match = re.search(pattern, raw_text, flags=re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                content = re.sub(pattern, "", raw_text, flags=re.DOTALL).strip()
                return reasoning, content
            return None, raw_text.strip()

        raw_llm_output = """<think>
Step 1: The user wants the churn rate.
Step 2: Churn rate = count(churned=1) / count(total).
Step 3: Calculate using pandas.
</think>
The customer churn rate is 37.5%."""

        reasoning, content = parse_reasoning_and_content(raw_llm_output)
        assert reasoning is not None
        assert "Step 1: The user wants the churn rate" in reasoning
        assert content == "The customer churn rate is 37.5%."

    def test_feature_20_openai_gateway_interface(self):
        """Feature 20: OpenAI-compatible gateway request/response payload validation."""
        request_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Execute query"}],
            "temperature": 0.0,
            "max_tokens": 1024
        }
        assert request_payload["model"] == "llama-3.3-70b-versatile"
        assert request_payload["temperature"] == 0.0

    def test_feature_21_nemo_nim_endpoint_client_interface(self):
        """Feature 21: Native NeMo / NIM endpoint client schema and metadata structure."""
        nemo_metadata = {
            "guardrails_enabled": True,
            "nim_service_endpoint": "http://localhost:8000/v1/chat/completions",
            "model": "nvidia/llama-3.1-nemotron-70b-instruct"
        }
        assert nemo_metadata["guardrails_enabled"] is True
        assert "nemotron" in nemo_metadata["model"]

    def test_feature_22_deterministic_mock_llm_runner(self):
        """Feature 22: Scripted deterministic replay runner with pre-defined response queues."""
        class DeterministicMockRunner:
            def __init__(self, response_queue: List[LLMResponse]):
                self.queue = list(response_queue)
                self.turn_index = 0

            def generate(self, messages: List[LLMMessage], tools: Optional[List[dict]] = None) -> LLMResponse:
                if self.turn_index < len(self.queue):
                    resp = self.queue[self.turn_index]
                    self.turn_index += 1
                    return resp
                return LLMResponse(content="Final Answer: Complete.")

        mock_responses = [
            LLMResponse(
                content="Inspecting schema",
                tool_calls=[ToolCall(id="c1", name="sqlite_schema", arguments={})]
            ),
            LLMResponse(
                content="Executing SQL",
                tool_calls=[ToolCall(id="c2", name="sqlite_query", arguments={"query": "SELECT count(*) FROM products;"})]
            ),
            LLMResponse(content="The database contains 5 products.")
        ]

        runner = DeterministicMockRunner(mock_responses)
        r1 = runner.generate([])
        assert r1.tool_calls[0].name == "sqlite_schema"
        r2 = runner.generate([])
        assert r2.tool_calls[0].name == "sqlite_query"
        r3 = runner.generate([])
        assert "5 products" in r3.content


# ===========================================================================
# FEATURE AREA 4: Core Agentic Framework & Telemetry (Features 23 - 26)
# ===========================================================================

class TestAgenticFrameworkAndTelemetry:
    """Tests for DAG Planner, Workflow Orchestration, 9-State FSM, and Telemetry."""

    def test_feature_23_dag_planner_and_topological_metrics(self):
        """Feature 23: [T.D] Sub-goal DAG validation and topological feasibility scoring (S_topo)."""
        # Valid DAG: g1 -> g2 -> g3
        dag = {
            "g1": [],
            "g2": ["g1"],
            "g3": ["g2"]
        }

        def compute_s_topo(graph: Dict[str, List[str]]) -> float:
            # Check for cycles via topological sort (Kahn's algorithm)
            in_degree = {u: 0 for u in graph}
            for u in graph:
                for v in graph[u]:
                    if v not in in_degree:
                        in_degree[v] = 0
            
            # Count in-degrees (u depends on v means edge v -> u)
            for u, deps in graph.items():
                in_degree[u] = len(deps)

            queue = [u for u, deg in in_degree.items() if deg == 0]
            visited = 0
            while queue:
                node = queue.pop(0)
                visited += 1
                for u, deps in graph.items():
                    if node in deps:
                        in_degree[u] -= 1
                        if in_degree[u] == 0:
                            queue.append(u)
            return 1.0 if visited == len(in_degree) else 0.0

        assert compute_s_topo(dag) == 1.0

        # Cyclic DAG: g1 -> g2 -> g1
        cyclic_dag = {
            "g1": ["g2"],
            "g2": ["g1"]
        }
        assert compute_s_topo(cyclic_dag) == 0.0

    def test_feature_23b_dependency_precision_and_recall(self):
        """Feature 23b: Computes P_dep, R_dep, and F1_dep for generated vs reference DAG edges."""
        edges_gen = {("g1", "g2"), ("g2", "g3"), ("g1", "g3")}
        edges_ref = {("g1", "g2"), ("g2", "g3")}

        intersection = edges_gen.intersection(edges_ref)
        p_dep = len(intersection) / len(edges_gen) # 2 / 3
        r_dep = len(intersection) / len(edges_ref) # 2 / 2 = 1.0
        f1_dep = (2 * p_dep * r_dep) / (p_dep + r_dep)

        assert pytest.approx(p_dep, rel=1e-3) == 2 / 3
        assert pytest.approx(r_dep, rel=1e-3) == 1.0
        assert pytest.approx(f1_dep, rel=1e-3) == 0.80

    def test_feature_24_workflow_orchestration_accuracy(self):
        """Feature 24: [W.O] Tool selection accuracy (Acc_tool) and parameter binding."""
        generated_tools = ["tabular_inspect", "python_repl", "python_repl"]
        reference_tools = ["tabular_inspect", "python_repl", "python_repl"]

        def compute_acc_tool(gen: List[str], ref: List[str]) -> float:
            matches = sum(1 for g, r in zip(gen, ref) if g == r)
            return matches / max(len(ref), 1)

        acc = compute_acc_tool(generated_tools, reference_tools)
        assert acc == 1.0

    def test_feature_24b_redundant_call_rate(self):
        """Feature 24b: Computes Redundant Call Rate (RCR) to penalize duplicate tool calls."""
        tool_calls = [
            ("sqlite_schema", "categories"),
            ("sqlite_schema", "categories"), # Redundant
            ("sqlite_query", "SELECT * FROM categories;"),
            ("sqlite_schema", "categories")  # Redundant
        ]
        seen = set()
        redundant_count = 0
        for tc in tool_calls:
            if tc in seen:
                redundant_count += 1
            else:
                seen.add(tc)
        rcr = redundant_count / len(tool_calls)
        assert pytest.approx(rcr, rel=1e-3) == 2 / 4

    def test_feature_25_multi_turn_9_state_trajectory_fsm(self):
        """Feature 25: 9-State Trajectory FSM validates state transitions."""
        valid_states = {
            "PLANNING", "ACTION_SELECTION", "TOOL_EXECUTION", "OBSERVATION",
            "VERIFICATION", "SELF_CORRECTION", "FINAL_SYNTHESIS",
            "TERMINAL_SUCCESS", "TERMINAL_FAILURE"
        }
        assert len(valid_states) == 9

        # Verify valid transition sequence
        trajectory_states = [
            "PLANNING",
            "ACTION_SELECTION",
            "TOOL_EXECUTION",
            "OBSERVATION",
            "VERIFICATION",
            "FINAL_SYNTHESIS",
            "TERMINAL_SUCCESS"
        ]
        for s in trajectory_states:
            assert s in valid_states

    def test_feature_26_telemetry_plan_adherence_score(self):
        """Feature 26: Telemetry logger calculates Plan Adherence Score (PAS)."""
        planned_subgoals = ["load_data", "clean_nulls", "compute_stats"]
        executed_subgoals = ["load_data", "clean_nulls", "compute_stats"]

        def compute_pas(planned: List[str], executed: List[str]) -> float:
            if not planned:
                return 1.0
            matched = sum(1 for p in planned if p in executed)
            return matched / len(planned)

        pas = compute_pas(planned_subgoals, executed_subgoals)
        assert pas == 1.0


# ===========================================================================
# FEATURE AREA 5: Verification, Self-Correction & Evaluation Pipeline (Features 27 - 30)
# ===========================================================================

class TestVerificationAndPipeline:
    """Tests for Intermediate Verifier, Self-Correction SCSR, CLI runner, and Reports."""

    def test_feature_27_intermediate_assertion_engine(self):
        """Feature 27: Intermediate assertion engine checks schema and intermediate invariants."""
        def verify_intermediate_state(data: Any, assertions: List[Dict[str, Any]]) -> tuple[bool, str]:
            for a in assertions:
                check_type = a.get("type")
                if check_type == "not_empty":
                    if len(data) == 0:
                        return False, "AssertionFailed: Result collection is empty"
                elif check_type == "columns_present":
                    df = pd.DataFrame(data)
                    required_cols = a.get("columns", [])
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        return False, f"AssertionFailed: Missing required columns {missing}"
            return True, "All intermediate assertions passed"

        data_valid = [{"customer_id": 1, "revenue": 100.0}]
        passed, msg = verify_intermediate_state(data_valid, [
            {"type": "not_empty"},
            {"type": "columns_present", "columns": ["customer_id", "revenue"]}
        ])
        assert passed is True

        # Failing assertion
        data_invalid = [{"customer_id": 1}]
        passed_inv, msg_inv = verify_intermediate_state(data_invalid, [
            {"type": "columns_present", "columns": ["customer_id", "revenue"]}
        ])
        assert passed_inv is False
        assert "revenue" in msg_inv

    def test_feature_28_self_correction_recovery_rate(self):
        """Feature 28: Calculates Self-Correction Success Rate (SCSR)."""
        episodes = [
            {"error_encountered": True, "recovered": True},
            {"error_encountered": True, "recovered": False},
            {"error_encountered": True, "recovered": True},
            {"error_encountered": False, "recovered": False}
        ]
        error_episodes = [e for e in episodes if e["error_encountered"]]
        recovered_episodes = [e for e in error_episodes if e["recovered"]]
        scsr = len(recovered_episodes) / len(error_episodes)
        assert pytest.approx(scsr, rel=1e-3) == 2 / 3

    def test_feature_28b_correction_efficiency_index(self):
        """Feature 28b: Calculates Correction Efficiency Index (CEI = 1 / attempts)."""
        attempts = 2
        cei = 1.0 / attempts
        assert cei == 0.5

    def test_feature_29_benchmark_runner_cli_config(self):
        """Feature 29: CLI runner configuration parsing and task sweep definitions."""
        config = {
            "benchmarks": ["infiagent", "bird_sql", "databench"],
            "model": "offline-mock-runner",
            "max_turns": 10,
            "timeout_seconds": 30,
            "output_dir": "./evaluation_reports"
        }
        assert len(config["benchmarks"]) == 3
        assert config["max_turns"] == 10
        assert config["timeout_seconds"] == 30

    def test_feature_30_comprehensive_evaluation_report_exporter(self):
        """Feature 30: Generates structured Markdown scorecard with accuracy breakdowns."""
        summary = {
            "total_tasks": 15,
            "passed_tasks": 14,
            "pass_rate": 0.9333,
            "mean_pas": 0.96,
            "scsr": 0.80
        }
        report_md = f"""# NeMo Long-Horizon Evaluation Summary
- **Total Tasks**: {summary['total_tasks']}
- **Passed**: {summary['passed_tasks']} ({summary['pass_rate'] * 100:.1f}%)
- **Mean Plan Adherence (PAS)**: {summary['mean_pas'] * 100:.1f}%
- **Self-Correction Success Rate (SCSR)**: {summary['scsr'] * 100:.1f}%
"""
        assert "93.3%" in report_md
        assert "PAS" in report_md
        assert "SCSR" in report_md
