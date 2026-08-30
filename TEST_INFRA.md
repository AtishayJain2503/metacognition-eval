# E2E Test Infrastructure & Test Suite Specification
# NVIDIA NeMo Long-Horizon Reasoning & Agent Evaluation Benchmark Harness

**Document Version**: 1.0.0  
**Target Project**: NeMo Long-Horizon Evaluation Harness (`nemo_long_horizon_eval`)  
**Workspace Root**: `c:\Projects\MetaCognition`  
**Test Suite Root**: `tests/e2e/`  
**Status**: ACTIVE / TEST READY  

---

## 1. Test Philosophy & Core Principles

The NeMo Long-Horizon Agent Evaluation Harness is designed to stress-test LLM agents across complex multi-step reasoning, task decomposition, workflow orchestration, verification, and self-correction using static local auxiliary tools with 0% network or browser dependency.

The E2E Testing Track adheres to five non-negotiable principles:

1. **Opaque-Box / Black-Box Requirement-Driven Testing**: Tests interact strictly through public interfaces, standard input/output envelopes (`ToolResult`, `BenchmarkTask`, `LLMResponse`, `EpisodeTrajectory`), CLI commands, and observable side-effects. Tests do not inspect internal private variables or depend on implementation internals.
2. **100% Hermetic & Offline-First Execution**: Zero external network requests, zero browser automation dependencies (Playwright/Selenium), and zero cloud API mandates during test execution. All datasets, SQLite databases, and model responses have deterministic offline fixtures or mocks.
3. **Deterministic Oracle Derivation**: Every test case derives its expected values from authoritative mathematical properties, static golden dataset references, or verified reference oracle executions with strict tolerance bounds ($\epsilon=0.01$, $\delta=0.01$).
4. **Self-Contained & Isolated Execution**: Every test sets up its own isolated environment (e.g. `:memory:` SQLite databases, isolated temp directories, fresh REPL state, seeded random states) and cleans up fully upon completion. No test depends on execution order.
5. **Progressive Testability & Graceful Adaptation**: Tests are structured to test interface contracts and runtime behaviors. Where components are invoked, contracts are validated against the Pydantic schemas and interface specifications defined in `PROJECT.md`.

---

## 2. Test Architecture & Runner Command

### 2.1 Directory Structure

```
tests/
└── e2e/
    ├── __init__.py
    ├── conftest.py                     # Shared fixtures (isolated DBs, synthetic tables, mock models)
    ├── test_tier1_features.py          # Tier 1: Feature unit/contract verification (≥5 per feature area)
    ├── test_tier2_boundaries.py        # Tier 2: Boundary conditions, errors, timeouts, security jailbreaks
    ├── test_tier3_pairwise.py          # Tier 3: Cross-feature combinations and multi-component interactions
    ├── test_tier4_scenarios.py          # Tier 4: Multi-step real-world application benchmark episodes
    └── test_tier5_adversarial.py       # Tier 5: Stress, mutation, and adversarial edge hardening (Phase 2)
```

### 2.2 Test Runner Invocations

The test suite can be run using `pytest`:

```powershell
# Run the entire E2E test suite (Tiers 1-4)
pytest tests/e2e -v

# Run individual tiers
pytest tests/e2e/test_tier1_features.py -v
pytest tests/e2e/test_tier2_boundaries.py -v
pytest tests/e2e/test_tier3_pairwise.py -v
pytest tests/e2e/test_tier4_scenarios.py -v

# Run with summary report
pytest tests/e2e -v --tb=short
```

---

## 3. Feature Inventory Coverage Matrix (All 32 Features)

The following matrix maps all 32 features defined in `PROJECT.md` to their corresponding milestone, test tiers, test files, and verification targets:

| # | Feature | Milestone | Test Tier | Test File | Target Verification Contract |
|---|---------|-----------|-----------|-----------|------------------------------|
| 1 | AST Security Validator | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Disallows `os`, `sys`, `socket`, `subprocess`, `__subclasses__`, `eval`, `exec` in AST |
| 2 | Dual-Phase REPL Compilation | M1 | Tier 1 | `test_tier1_features.py` | Splits statements into `exec` body and `eval` tail; captures exact expression return value |
| 3 | Safe Builtins Whitelist | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Curates `__builtins__` namespace; blocks `open()`, `__import__()`, `breakpoint()` |
| 4 | REPL Stream Redirection & Hard Timeout | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Captures stdout/stderr via `io.StringIO`; terminates worker process on timeout expiry |
| 5 | Transient SQLite Lifecycle | M1 | Tier 1 | `test_tier1_features.py` | In-memory `:memory:` and temp disk lifecycles with automated DDL/DML seeding and teardown |
| 6 | Database Read-Only PRAGMA & Progress Handlers | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Enforces `PRAGMA query_only = ON` and opcode progress handler abortion on infinite CTE loops |
| 7 | SQLite Schema Introspection Engine | M1 | Tier 1 | `test_tier1_features.py` | Extracts tables, column dtypes, nullability, primary/foreign keys, and 3-row data previews |
| 8 | SQL Query Result Bounding & Pagination | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Caps query output to 50 rows by default, provides `has_more` flag and pagination hints |
| 9 | Tabular Columnar Schema & Profiler | M1 | Tier 1 | `test_tier1_features.py` | Reads CSV/Parquet, inspects dtypes, missingness, and outputs 8-point numerical/categorical stats |
| 10 | Diagnostic Error Formatter & Remediation | M1 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Visual caret pointing to error line/col, categorizes error types, provides actionable hints |
| 11 | Tool JSON-Schema Definitions | M1 | Tier 1 | `test_tier1_features.py` | OpenAI/NeMo function schemas for `python_repl`, `sqlite_query`, `sqlite_schema`, `tabular_inspect` |
| 12 | InfiAgent-DABench Task Ingestion | M2 | Tier 1 | `test_tier1_features.py` | Ingests analytics tasks, context schemas, code blocks, and regex answer delimiter extraction |
| 13 | BIRD-SQL / Spider 2.0-Lite Ingestion | M2 | Tier 1 | `test_tier1_features.py` | Ingests text-to-SQL tasks, golden queries, DDL schemas, and domain evidence dictionaries |
| 14 | DataBench Ingestion & Categorization | M2 | Tier 1 | `test_tier1_features.py` | Ingests tabular tasks across 4 semantic types (Scalar, Boolean, List/Set, Table) |
| 15 | Deterministic Offline Synthetic Fixtures | M2 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Generates hermetic relational SQLite databases and CSV/Parquet datasets for offline CI |
| 16 | Polymorphic Ground Truth Evaluation Engine | M2 | Tier 1, 2 | `test_tier1_features.py`, `test_tier2_boundaries.py` | Exact string/scalar, float tolerance ($\epsilon=0.01$), multiset SQL match, DataFrame diffing |
| 17 | Pass@k Unbiased Metric Estimator | M2 | Tier 1 | `test_tier1_features.py` | Computes unbiased pass@k accuracy $1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ across sample sweeps |
| 18 | Unified BaseLLMClient Protocol | M3 | Tier 1 | `test_tier1_features.py` | Abstract provider interface with sync `generate()` and async `agenerate()` supporting tool calls |
| 19 | Groq API Provider Client | M3 | Tier 1 | `test_tier1_features.py` | High-throughput client with `<think>` tag extraction and reasoning content isolation |
| 20 | OpenAI-Compatible Gateway Client | M3 | Tier 1 | `test_tier1_features.py` | Standard HTTP client for `/v1/chat/completions` endpoints (vLLM, Ollama, TGI) |
| 21 | Native NeMo / NIM Endpoint Client | M3 | Tier 1 | `test_tier1_features.py` | Connector for NVIDIA NeMo microservices and NeMo Guardrail / Agent endpoints |
| 22 | Deterministic Offline Mock LLM Runner | M3 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Scripted replay runner with configurable response queues and deterministic error injection |
| 23 | Task Decomposition & Planning Module [T.D] | M4 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Sub-goal DAG generation, topological validation, and plan quality scoring ($S_{topo}, P_{dep}$) |
| 24 | Workflow Orchestration Engine [W.O] | M4 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Tool selection precision, parameter binding across turns, and chaining ($Acc_{tool}, SPEA$) |
| 25 | Multi-Turn 9-State Trajectory FSM | M4 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Tracks 9 discrete states (`PLANNING` $\to$ `ACTION` $\to$ `TOOL` $\to$ `VERIF` $\to$ `CORRECT` $\to$ `FINAL`) |
| 26 | Multi-Turn Telemetry & Trace Exporter | M4 | Tier 1, 3 | `test_tier1_features.py`, `test_tier3_pairwise.py` | Records step events, Plan Adherence Score ($PAS$), and exports JSONL / OTLP / Markdown |
| 27 | Intermediate Assertion & Verification Engine | M5 | Tier 1, 4 | `test_tier1_features.py`, `test_tier4_scenarios.py` | Schema/shape sanity checks, self-assessment assertions, and intermediate state validators |
| 28 | Iterative Improvement & Self-Correction Loop | M5 | Tier 1, 4 | `test_tier1_features.py`, `test_tier4_scenarios.py` | Intercepts tool errors, generates visual remediation prompts, and measures $SCSR$ / $TOP$ |
| 29 | Benchmark Runner CLI & Pipeline | M5 | Tier 1, 4 | `test_tier1_features.py`, `test_tier4_scenarios.py` | CLI orchestrator executing multi-dataset, multi-model evaluation sweeps with seed configs |
| 30 | Comprehensive Evaluation Report Exporter | M5 | Tier 1, 4 | `test_tier1_features.py`, `test_tier4_scenarios.py` | Compiles aggregate pass rates, radar chart metrics, tool validity indices, and error logs |
| 31 | Full E2E Test Suite Pass (Tiers 1-4) | M6 | Tiers 1-4 | All test files | 100% execution pass rate across all feature, boundary, pairwise, and scenario suites |
| 32 | Adversarial Coverage Hardening (Tier 5) | M6 | Tier 5 | `test_tier5_adversarial.py` | Mutation testing, complex AST obfuscation bypasses, extreme load, and corruption recovery |

---

## 4. Real-World Application Scenarios (Tier 4)

Tier 4 tests model end-to-end multi-step reasoning episodes mirroring production enterprise data analysis workflows:

| Scenario ID | Name | Benchmark Archetype | Tools Involved | Horizon (Turns) | Key Verification Invariants |
|-------------|------|---------------------|----------------|-----------------|-----------------------------|
| `SCENARIO-01` | End-to-End Customer Churn Analytics Pipeline | InfiAgent-DABench | `tabular_inspect`, `python_repl` | 4–6 turns | Tabular ingestion, missing value imputation, multi-variable correlation computation, top-3 feature extraction, exact float matching |
| `SCENARIO-02` | Multi-Turn Text-to-SQL Join & Aggregation with Disambiguation | BIRD-SQL / Spider 2.0 | `sqlite_schema`, `sqlite_query` | 3–5 turns | Schema introspection of foreign keys, domain evidence disambiguation, compound SQL query execution, multiset row verification |
| `SCENARIO-03` | Iterative Error Recovery & Code Self-Correction Episode | InfiAgent-DABench | `python_repl`, `diagnostics` | 3–4 turns | Syntax/Runtime error generation (KeyError/ZeroDivisionError), diagnostic hint injection, automated code repair, final answer correctness |
| `SCENARIO-04` | Tabular Semantic Reasoning & Hybrid Type Resolution | DataBench | `tabular_inspect`, `python_repl` | 3–5 turns | Ingestion of mixed-type columns (currency strings, dates), type-normalized filtering, multi-category aggregation, exact boolean/scalar check |
| `SCENARIO-05` | Full Evaluation Harness CLI Multi-Task Evaluation Sweep | Evaluation Pipeline | All tools, `MockLLM`, `FSM`, `Pipeline` | Multi-task batch | End-to-end evaluation run across 3 synthetic benchmarks, trajectory recording ($PAS \ge 0.90$), and Markdown/JSON report export |

---

## 5. Coverage Thresholds & Quality Gates

To ensure thorough coverage across the evaluation framework, the test suite enforces the following quantitative thresholds:

1. **Tier 1 (Feature Contracts)**: $\ge 5$ distinct test cases per feature area:
   - Tool sandboxes & auxiliary engines ($\ge 5$ tests)
   - Dataset ingestion & ground truth evaluation ($\ge 5$ tests)
   - Model provider interfaces & mock runners ($\ge 5$ tests)
   - Agent planning, orchestration & telemetry ($\ge 5$ tests)
   - Verification, self-correction & pipeline CLI ($\ge 5$ tests)
   - **Total Tier 1 Target**: $\ge 25$ tests.
2. **Tier 2 (Boundaries & Edge Cases)**: $\ge 5$ boundary test cases per domain:
   - REPL AST security jailbreaks & forbidden imports/attributes ($\ge 5$)
   - Hard execution timeouts & runaway loops/CTEs ($\ge 5$)
   - Numerical tolerances, zero division, NaN, infinite floats ($\ge 5$)
   - Empty databases, schema mismatches, malformed CSVs ($\ge 5$)
   - **Total Tier 2 Target**: $\ge 20$ tests.
3. **Tier 3 (Pairwise Interactions)**:
   - Python REPL $\times$ Tabular CSV/Parquet Dataframes
   - SQLite Engine $\times$ Diagnostic Error Remediation
   - Mock LLM Client $\times$ 9-State Trajectory FSM & Telemetry
   - Plan Adherence Scorer $\times$ Orchestration Parameter Bridging
   - **Total Tier 3 Target**: $\ge 10$ tests.
4. **Tier 4 (Real-World Scenarios)**:
   - $\ge 5$ comprehensive, multi-step scenarios executing end-to-end.
5. **Determinism & Flakiness Policy**:
   - Zero non-deterministic sleep commands.
   - All random seeds explicitly fixed (`seed=42`).
   - SQLite queries explicitly ordered (`ORDER BY`) or multiset-compared.
   - Timers use monotonic clocks or mock clock injections.
