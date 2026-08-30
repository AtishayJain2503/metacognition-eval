# Project: NeMo Long-Horizon Reasoning & Agent Evaluation Benchmark Harness

## Architecture

The system is a deterministic, hermetic, and offline-first evaluation harness for NVIDIA NeMo agent architectures. It stress-tests LLM agents across long-horizon multi-step reasoning, task decomposition, workflow orchestration, verification, and self-correction using static local auxiliary tools with 0% network or browser dependency.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    CLI / Runner                                        │
│               (Configuration, Execution Pipeline, Benchmark Orchestration)             │
└───────────────┬────────────────────────────────────────────────────────┬───────────────┘
                │                                                        │
                ▼                                                        ▼
┌────────────────────────────────┐                     ┌─────────────────────────────────┐
│     Model Provider Layer       │                     │    Benchmark Dataset Ingestion  │
│  - Groq API Client             │                     │  - InfiAgent-DABench / DAEval   │
│  - OpenAI-Compatible Gateway   │                     │  - BIRD-SQL / Spider 2.0-Lite   │
│  - Native NeMo / NIM Client    │                     │  - DataBench (CSV / Parquet)    │
│  - Offline Deterministic Mock  │                     │  - Offline Synthetic Fixtures   │
└───────────────┬────────────────┘                     └────────────────┬────────────────┘
                │                                                        │
                ▼                                                        ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Agentic Capability Execution Core                               │
│  - Task Decomposition & Planning [T.D] (DAG generation, topological scoring)           │
│  - Workflow Orchestration [W.O] (Tool selection, parameter bridging, chaining)         │
│  - Multi-Turn 9-State Trajectory FSM & Telemetry Logger (JSONL / OTLP)                 │
└────────────────┬───────────────────────────────────────────────────────┬───────────────┘
                 │                                                       │
                 ▼                                                       ▼
┌────────────────────────────────┐                     ┌─────────────────────────────────┐
│     Hermetic Auxiliary Tools   │                     │  Verification & Self-Correction │
│  - Python REPL Sandbox (AST)   │                     │  - Intermediate Assertions      │
│  - SQLite Database Engine      │                     │  - Polymorphic Ground Truth     │
│  - Tabular Data (Pandas/Arrow) │                     │  - Iterative Error Recovery     │
│  - Diagnostic Error Formatter  │                     │  - Trajectory Scoring Metrics   │
└────────────────────────────────┘                     └─────────────────────────────────┘
```

---

## Feature Inventory

Every feature discovered during the survey phase is mapped to a specific milestone below:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | AST Security Validator | Static AST analysis prohibiting dangerous imports (`os`, `sys`, `socket`) and dunder traversal (`__subclasses__`) | M1 | Survey Tools |
| 2 | Dual-Phase REPL Compilation | Executes definitions as `exec` and terminal expressions as `eval` to capture exact return values | M1 | Survey Tools |
| 3 | Safe Builtins Whitelist | Restricts `__builtins__` namespace to safe mathematical, data structure, and exception primitives | M1 | Survey Tools |
| 4 | REPL Stream Redirection & Hard Timeout | Captures `stdout`/`stderr` and enforces process-isolated wall-clock timeout with clean worker termination | M1 | Survey Tools |
| 5 | Transient SQLite Lifecycle | Spins up `:memory:` and temporary SQLite instances with seed schema/data and clean teardown | M1 | Survey Tools |
| 6 | Database Read-Only PRAGMA & Progress Handlers | Enforces `PRAGMA query_only = ON` and CPU opcode progress handler timeouts against runaway CTEs | M1 | Survey Tools |
| 7 | SQLite Schema Introspection Engine | Extracts tables, column types, nullability, primary/foreign keys, and sample rows | M1 | Survey Tools |
| 8 | SQL Query Result Bounding & Pagination | Limits query outputs to 50 rows by default with truncation flags and paging suggestions | M1 | Survey Tools |
| 9 | Tabular Columnar Schema & Profiler | Ingests CSV/Parquet, inspects dtypes, nulls, and generates numerical/categorical summary statistics | M1 | Survey Tools |
| 10 | Diagnostic Error Formatter & Remediation | Transforms Python/SQL exceptions into visual caret pointers and actionable hints (`Did you mean ...?`) | M1 | Survey Tools |
| 11 | Tool JSON-Schema Definitions | OpenAI/NeMo compatible function-calling schemas for `python_repl`, `sqlite_query`, `sqlite_schema`, `tabular_inspect` | M1 | Survey Tools |
| 12 | InfiAgent-DABench Task Ingestion | Ingests data analytics tasks, environment definitions, and regex answer delimiter extraction | M2 | Survey Datasets |
| 13 | BIRD-SQL / Spider 2.0-Lite Ingestion | Ingests relational text-to-SQL tasks, golden queries, DDL schemas, and domain evidence dictionaries | M2 | Survey Datasets |
| 14 | DataBench Ingestion & Categorization | Ingests tabular reasoning tasks across 4 semantic types (Scalar, Boolean, List/Set, Table) | M2 | Survey Datasets |
| 15 | Deterministic Offline Synthetic Fixtures | Built-in offline SQLite relational databases and tabular datasets for 100% hermetic CI/CD testing | M2 | Survey Datasets |
| 16 | Polymorphic Ground Truth Evaluation Engine | Exact match, float tolerance ($\epsilon=0.01$, $\delta=0.01$), multiset SQL execution match, DataFrame diffing | M2 | Survey Datasets |
| 17 | Pass@k Unbiased Metric Estimator | Computes unbiased pass@k accuracy estimators across multi-sample evaluation runs | M2 | Survey Datasets |
| 18 | Unified BaseLLMClient Protocol | Abstract provider interface with sync `generate()` and async `agenerate()` supporting tool calls and streaming | M3 | Survey Architecture |
| 19 | Groq API Provider Client | Ultra-fast inference for `llama-3.3-70b-versatile` and `deepseek-r1-distill-llama-70b` with `<think>` isolation | M3 | Survey Architecture |
| 20 | OpenAI-Compatible Gateway Client | Generic HTTP client for v1/chat/completions compatible local endpoints (vLLM, Ollama, TGI) | M3 | Survey Architecture |
| 21 | Native NeMo / NIM Endpoint Client | Interface for NVIDIA NeMo microservices and NeMo Guardrail / Agent endpoints | M3 | Survey Architecture |
| 22 | Deterministic Offline Mock LLM Runner | Offline scripted replay runner with configurable deterministic error injection for CI/CD test suites | M3 | Survey Architecture |
| 23 | Task Decomposition & Planning Module [T.D] | Sub-goal DAG generation, topological dependency validation, and scoring ($S_{topo}, P_{dep}, S_{struct}$) | M4 | Survey Architecture |
| 24 | Workflow Orchestration Engine [W.O] | Tool selection accuracy, cross-tool parameter bridging, and execution chaining ($Acc_{tool}, SPEA$) | M4 | Survey Architecture |
| 25 | Multi-Turn 9-State Trajectory FSM | Finite State Machine tracking agent lifecycle (`PLANNING` $\to$ `ACTION` $\to$ `TOOL` $\to$ `VERIF` $\to$ `CORRECT` $\to$ `FINAL`) | M4 | Survey Architecture |
| 26 | Multi-Turn Telemetry & Trace Exporter | Step-level event logger recording plan adherence ($PAS$), tool call validity, JSONL stream & OTLP spans | M4 | Survey Architecture |
| 27 | Intermediate Assertion & Verification Engine | Schema/shape consistency checks, self-assessment validators, and intermediate assertion testing | M5 | Survey Architecture |
| 28 | Iterative Improvement & Self-Correction Loop | Intercepts runtime exceptions, generates remediation prompts, and tracks recovery metrics ($SCSR, CEI, TOP$) | M5 | Survey Architecture |
| 29 | Benchmark Runner CLI & Pipeline | Configurable CLI orchestration pipeline executing multi-dataset, multi-model evaluation sweeps | M5 | Survey Architecture |
| 30 | Comprehensive Evaluation Report Exporter | Generates structured Markdown and JSON telemetry summaries with score breakdowns and failure traces | M5 | Survey Architecture |
| 31 | Full E2E Test Suite Pass (Tiers 1-4) | 100% pass on comprehensive E2E test suite covering features, boundaries, pairwise, and real-world scenarios | M6 | Survey / Dual Track |
| 32 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing and mutation verification to eliminate all latent bugs and edge gaps | M6 | Survey / Dual Track |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Hermetic Tool Sandboxes & Auxiliary Engines | Python REPL AST sandbox, SQLite engine & progress handlers, Tabular engine, diagnostic error formatter, tool JSON schemas | None | DONE |
| M2 | Benchmark Dataset Ingestion & Ground Truth Engines | InfiAgent, BIRD-SQL, DataBench loaders, offline synthetic fixtures, polymorphic evaluation engine (Exact, Float, SQL Multiset, DataFrame diffing) | None | DONE |
| M3 | Model Provider & Inference Interfaces | `BaseLLMClient`, Groq client with `<think>` isolation, OpenAI gateway, NeMo/NIM client, Offline Deterministic Mock Runner | None | DONE |
| M4 | Core Agentic Framework & Multi-Turn Orchestrator | [T.D] DAG planner & topological metrics, [W.O] tool orchestration, 9-state FSM trajectory machine, multi-turn telemetry logger ($PAS$) | M1, M3 | PLANNED |
| M5 | Verification, Self-Correction & Benchmark Pipeline | Intermediate assertion checks, iterative error recovery loop ($SCSR$), Benchmark Runner CLI, multi-dataset evaluation reports | M1, M2, M3, M4 | PLANNED |
| M6 | Final Integration, 100% E2E Verification & Tier 5 Hardening | Pass 100% of E2E test suite (Tiers 1-4) from E2E Track, followed by Phase 2 Tier 5 adversarial hardening | M1, M2, M3, M4, M5 | PLANNED |

---

## Interface Contracts

### 1. `nemo_eval.tools` ↔ `nemo_eval.agents`
```python
class ToolResult(BaseModel):
    status: Literal["success", "error"]
    execution_time_ms: float
    data: Any = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[DiagnosticError] = None

class DiagnosticError(BaseModel):
    error_type: str
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: str
    raw_traceback: str
```

### 2. `nemo_eval.datasets` ↔ `nemo_eval.eval`
```python
class BenchmarkTask(BaseModel):
    task_id: str
    benchmark_name: Literal["infiagent", "bird_sql", "databench", "synthetic"]
    query: str
    context_schema: Optional[Dict[str, Any]] = None
    db_path: Optional[str] = None
    table_path: Optional[str] = None
    ground_truth: Any
    eval_type: Literal["exact", "float_tol", "sql_multiset", "dataframe_diff"]
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 3. `nemo_eval.models` ↔ `nemo_eval.agents`
```python
class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class LLMResponse(BaseModel):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None # Isolated <think> content for DeepSeek-R1
    tool_calls: List[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: Optional[Dict[str, Any]] = None
```

### 4. `nemo_eval.telemetry` ↔ `nemo_eval.correction` / `nemo_eval.pipeline`
```python
class StepEvent(BaseModel):
    step_id: int
    state: Literal["PLANNING", "ACTION_SELECTION", "TOOL_EXECUTION", "OBSERVATION", "VERIFICATION", "SELF_CORRECTION", "FINAL_SYNTHESIS", "TERMINAL_SUCCESS", "TERMINAL_FAILURE"]
    timestamp: float
    duration_ms: float
    input_payload: Dict[str, Any]
    output_payload: Dict[str, Any]
    metrics: Dict[str, float] = Field(default_factory=dict) # e.g. plan_adherence, tool_validity

class EpisodeTrajectory(BaseModel):
    task_id: str
    model_name: str
    status: Literal["success", "failed", "timeout", "max_turns_exceeded"]
    steps: List[StepEvent]
    total_duration_ms: float
    plan_adherence_score: float
    self_correction_attempts: int
    self_correction_success: bool
    final_answer: Any
    ground_truth_score: float
```

---

## Code Layout

```
c:/Projects/MetaCognition/
├── nemo_eval/
│   ├── __init__.py
│   ├── cli.py                          # CLI runner for evaluation pipeline
│   ├── tools/                          # Milestone 1 (Hermetic Auxiliary Tools)
│   │   ├── __init__.py
│   │   ├── repl.py                     # Python REPL sandbox, AST validator, process worker
│   │   ├── sqlite_engine.py            # SQLite lifecycle, progress handlers, schema inspector
│   │   ├── tabular.py                  # CSV/Parquet parsers, summarizer, profiler
│   │   ├── diagnostics.py              # Error classifier, syntax highlighter, hints
│   │   └── schemas.py                  # OpenAI/NeMo JSON-schemas & Pydantic definitions
│   ├── datasets/                       # Milestone 2 (Benchmark Datasets & Fixtures)
│   │   ├── __init__.py
│   │   ├── base.py                     # Dataset base class & schema loader
│   │   ├── infiagent.py                # InfiAgent-DABench parser & execution adapter
│   │   ├── bird_sql.py                 # BIRD-SQL / Spider 2.0-Lite loader & schemas
│   │   ├── databench.py                # DataBench tabular parser & semantic categorizer
│   │   └── synthetic.py                # Hermetic offline synthetic fixture generator
│   ├── eval/                           # Milestone 2 (Ground Truth Evaluation)
│   │   ├── __init__.py
│   │   ├── engine.py                   # Polymorphic evaluation router
│   │   ├── exact.py                    # Exact & normalized string/scalar matching
│   │   ├── numerical.py                # Dual relative/absolute float tolerance
│   │   ├── sql_match.py                # Multiset Counter SQL execution equivalence
│   │   ├── dataframe_diff.py           # Polars/Pandas DataFrame diffing engine
│   │   └── metrics.py                  # Pass@k unbiased estimator, aggregate scorecards
│   ├── models/                         # Milestone 3 (Model Provider Interfaces)
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseLLMClient abstract protocol
│   │   ├── groq.py                     # Groq API client with <think> isolation
│   │   ├── openai_gateway.py           # OpenAI-compatible HTTP client
│   │   ├── nemo_client.py              # Native NVIDIA NeMo / NIM client
│   │   └── mock_runner.py              # Deterministic offline mock replay & error injector
│   ├── agents/                         # Milestone 4 (Core Agentic Capabilities)
│   │   ├── __init__.py
│   │   ├── planner.py                  # [T.D] Sub-goal DAG generation & topological scoring
│   │   ├── orchestrator.py             # [W.O] Tool selection, parameter bridging, chaining
│   │   └── agent_loop.py               # Multi-turn reasoning agent execution loop
│   ├── telemetry/                      # Milestone 4 (Multi-Turn Telemetry)
│   │   ├── __init__.py
│   │   ├── tracer.py                   # 9-State Trajectory FSM and event logger
│   │   ├── metrics.py                  # Plan Adherence Score (PAS) and tool metrics
│   │   └── exporters.py                # JSONL, OTLP, and Markdown trajectory exporters
│   ├── correction/                     # Milestone 5 (Verification & Self-Correction)
│   │   ├── __init__.py
│   │   ├── verifier.py                 # Intermediate assertion and schema checker
│   │   └── self_correct.py             # Error recovery loop & SCSR / TOP calculator
│   └── pipeline/                       # Milestone 5 (Evaluation Pipeline)
│       ├── __init__.py
│       ├── runner.py                   # Multi-dataset, multi-model evaluation harness
│       ├── config.py                   # Pipeline configuration and YAML/JSON loader
│       └── reporter.py                 # Summary scorecard and failure analyzer
├── tests/
│   ├── unit/                           # Unit tests for isolated modules
│   └── e2e/                            # Opaque-box E2E test suite (Tiers 1-5)
│       ├── tier1_features/
│       ├── tier2_boundaries/
│       ├── tier3_pairwise/
│       ├── tier4_scenarios/
│       └── tier5_adversarial/
├── TEST_INFRA.md                       # E2E Test Suite Index & Architecture
├── TEST_READY.md                       # E2E Test Suite Readiness Signal
└── PROJECT.md                          # Global Architecture & Milestone Decomposition
```

### File Write Ownership & Boundaries
- Milestone 1 Sub-orchestrator exclusively owns `nemo_eval/tools/*`
- Milestone 2 Sub-orchestrator exclusively owns `nemo_eval/datasets/*` and `nemo_eval/eval/*`
- Milestone 3 Sub-orchestrator exclusively owns `nemo_eval/models/*`
- Milestone 4 Sub-orchestrator exclusively owns `nemo_eval/agents/*` and `nemo_eval/telemetry/*`
- Milestone 5 Sub-orchestrator exclusively owns `nemo_eval/correction/*`, `nemo_eval/pipeline/*`, and `nemo_eval/cli.py`
- E2E Testing Orchestrator exclusively owns `TEST_INFRA.md`, `TEST_READY.md`, and `tests/e2e/*`
