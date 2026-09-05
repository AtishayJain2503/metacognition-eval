# Project: metacognition-eval

## Architecture
`metacognition-eval` is an end-to-end mathematical and multi-domain reasoning benchmark harness supporting MATH (Hendrycks), PutnamBench, and Lila datasets, featuring dual-mode execution (Vanilla Zero-Shot vs 9-state Agentic FSM with Python REPL) and real-time hardware telemetry (execution time, peak RAM, GPU VRAM, power, and energy consumption).

### Subsystem Topology
```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 CLI Runner / Benchmark Engine           │
                  │   (--mode vanilla | agentic | both, 7 models, sweeps)   │
                  └────────────┬───────────────────────────────┬────────────┘
                               │                               │
            ┌──────────────────▼──────────┐         ┌──────────▼─────────────────┐
            │       Vanilla Engine        │         │        Agentic FSM         │
            │  (Zero-Shot CoT, No Tools)  │         │ (9-State FSM + Python REPL)│
            └──────────────────┬──────────┘         └──────────┬─────────────────┘
                               │                               │
                               ├───────────────────────────────┤
                               │                               │
                  ┌────────────▼───────────────────────────────▼────────────┐
                  │          Hardware Telemetry Monitor                     │
                  │ (psutil RAM, pynvml / nvidia-smi GPU & Power, Joules)  │
                  └────────────────────────────┬────────────────────────────┘
                                               │
            ┌──────────────────────────────────┼──────────────────────────────────┐
            ▼                                  ▼                                  ▼
┌───────────────────────┐          ┌───────────────────────┐          ┌───────────────────────┐
│     MATH Loader       │          │   PutnamBench Loader  │          │      Lila Loader      │
│ (50 Hendrycks Tasks)  │          │ (50 Competition Tasks)│          │(7 Categories x 50=350)│
└───────────┬───────────┘          └───────────┬───────────┘          └───────────┬───────────┘
            │                                  │                                  │
            └──────────────────────────────────┼──────────────────────────────────┘
                                               │
                                   ┌───────────▼───────────┐
                                   │  Polymorphic Evaluator│
                                   │  & ValueExtractor     │
                                   │ (LaTeX, SymPy, Floats)│
                                   └───────────┬───────────┘
                                               │
                                   ┌───────────▼───────────┐
                                   │  Reports & Scorecards │
                                   │ (results/*.md, .jsonl)│
                                   └───────────────────────┘
```

---

## Code Layout

```
C:\Projects\MetaCognition/
├── pyproject.toml
├── nemo_eval/
│   ├── __init__.py
│   ├── cli.py                        # Top-level CLI entry point
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── monitor.py                # HardwareMonitor (psutil, pynvml, energy_joules)
│   │   ├── tracer.py                 # 9-State FSM, StepEvent, EpisodeTrajectory
│   │   └── extractor.py              # ValueExtractor (LaTeX boxed, regex, numeric, SymPy)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_loop.py             # 9-State Agentic FSM loop with Python REPL
│   │   ├── vanilla.py                # VanillaEngine (Zero-Shot CoT, 0 tools)
│   │   └── planner.py                # DAG Task Planner
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── repl.py                   # AST-validated Python REPL sandbox
│   │   ├── sqlite_engine.py          # SQLite tool
│   │   └── tabular.py                # Tabular inspection tool
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseDatasetLoader & BenchmarkTask
│   │   ├── math.py                   # MATHLoader (50 samples)
│   │   ├── putnam.py                 # PutnamBenchLoader (50 competition tasks)
│   │   ├── lila.py                   # LilaLoader (7 subcategories x 50 = 350 tasks)
│   │   ├── fixtures/                 # Deterministic offline JSONL fixtures
│   │   └── gsm8k.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── engine.py                 # Polymorphic evaluation dispatcher
│   │   ├── math_eval.py              # SympyMathEvaluator (algebraic equivalence, LaTeX)
│   │   ├── exact.py
│   │   └── numerical.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseLLMClient protocol
│   │   ├── openai_client.py          # Ollama, vLLM, OpenAI Gateway
│   │   ├── groq_client.py            # Groq DeepSeek-R1 with think isolation
│   │   └── mock_client.py            # DeterministicMockLLMClient
│   └── pipeline/
│       ├── __init__.py
│       ├── runner.py                 # BenchmarkRunner & Dual-Mode sweep harness
│       └── reporter.py               # Markdown scorecard & comparison table generator
├── results/                          # Evaluation outputs, scorecards, telemetry logs
└── tests/
    ├── unit/
    └── e2e/
        ├── test_metacognition_eval_e2e.py
        └── test_harness.py
```

---

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | HardwareTelemetryMonitor | Background sampler for RAM via `psutil` and GPU VRAM & Power via `pynvml`/`nvidia-smi` | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Telemetry Step & Episode Schema | `StepEvent` and `EpisodeTrajectory` recording `duration_ms`, `peak_ram_mb`, `gpu_vram_mb`, `gpu_power_watts`, `energy_joules` | M1 | ORIGINAL_REQUEST §R1 |
| 3 | GPU Fallback Engine | Graceful degradation to 0.0 values with `gpu_available: False` when running in headless / non-NVIDIA environments | M1 | Survey |
| 4 | Sandboxed Subprocess Execution | Hard timeouts and isolated execution preventing infinite loops or memory leaks in tools | M1 | ORIGINAL_REQUEST §AC |
| 5 | Value-Only Answer Extractor | Tiered extraction: LaTeX `\boxed{}`, balanced braces, JSON values, regex anchors, numeric fallbacks, stripping prose | M1 | ORIGINAL_REQUEST §R1 |
| 6 | MATH (Hendrycks) Dataset Loader | Ingests 50 deterministic samples across subjects with LaTeX `\boxed{}` solutions | M2 | ORIGINAL_REQUEST §R3 |
| 7 | PutnamBench Dataset Loader | Ingests 50 competition-grade mathematical tasks with formal ground truths | M2 | ORIGINAL_REQUEST §R3 |
| 8 | Lila (AllenAI) Dataset Loader | Ingests 50 samples for each of the 7 core subcategories (350 total: Arithmetic, Algebra, Calculus, Geometry, Combinatorics, Physics, Statistics) | M2 | ORIGINAL_REQUEST §R3 |
| 9 | SympyMathEvaluator | Symbolic mathematical equivalence checker evaluating LaTeX and algebraic expressions ($|cand - gold| \le \epsilon + \delta |gold|$) | M2 | ORIGINAL_REQUEST §R3 |
| 10 | Polymorphic Ground Truth Evaluator | Multi-type evaluator supporting numbers, floats with tolerance, expressions, fractions, and sets | M2 | ORIGINAL_REQUEST §R3 |
| 11 | Offline Deterministic Fixtures | Hermetic local JSONL fixtures for MATH, Putnam, and Lila enabling 100% offline test execution | M2 | Survey |
| 12 | Vanilla Zero-Shot Engine | Pure zero-shot / single-shot Chain-of-Thought execution with zero tool access | M3 | ORIGINAL_REQUEST §R2 |
| 13 | 9-State Agentic FSM Engine | Multi-turn FSM (PLANNING, ACTION_SELECTION, TOOL_EXECUTION, OBSERVATION, VERIFICATION, SELF_CORRECTION, FINAL_SYNTHESIS, TERMINAL_SUCCESS, TERMINAL_FAILURE) with Python REPL | M3 | ORIGINAL_REQUEST §R2 |
| 14 | Identical Split Parity Guarantee | Dual-mode execution engine ensuring Vanilla and Agentic modes evaluate on the exact same task instances | M3 | ORIGINAL_REQUEST §R2 |
| 15 | Delta Performance Metrics | Automatic calculation of $\Delta \text{Acc}$, latency speedup/overhead, and energy ratio between modes | M3 | Survey |
| 16 | 7 Target Models Configuration | Support for `Qwen2.5-Math-7B`, `DeepSeek-R1-7B`, `Phi4-mini-reasoning`, `Llama3.2-3B`, `Qwen2.5-Math-1.5B`, `DeepSeek-R1-1.5B`, `Qwen3-4B-Thinking` | M4 | ORIGINAL_REQUEST §R4 |
| 17 | Multi-Model CLI Sweep Runner | Unified CLI commands (`--mode vanilla|agentic|both`, `--dataset math|putnam|lila|all`, `--models ...`) | M4 | ORIGINAL_REQUEST §R4 |
| 18 | Markdown Scorecard Generator | Auto-generation of `summary_scorecard.md`, dataset breakdowns, and accuracy leaderboards | M4 | ORIGINAL_REQUEST §R4 |
| 19 | Telemetry Comparison Tables | Side-by-side tables comparing accuracy, duration_ms, peak_ram_mb, gpu_vram_mb, energy_joules | M4 | ORIGINAL_REQUEST §R4 |
| 20 | Streaming JSONL Telemetry Traces | Comprehensive episode-by-episode JSONL output under `results/` | M4 | ORIGINAL_REQUEST §AC |
| 21 | E2E Test Suite Tiers 1-4 | Requirement-driven opaque-box test suite (Feature coverage, BVA, Pairwise, Real-world scenarios) | M5 | TEST_INFRA |
| 22 | Adversarial Hardening (Tier 5) | White-box adversarial testing, edge cases, stress tests, zero gaps | M5 | TEST_INFRA |
| 23 | 1,000 MATH Ingestion & Stratified Sampling | Stratified sample across 7 subjects and difficulty Levels 1–5 | M6 | ORIGINAL_REQUEST §R1 |
| 24 | 1,000 PutnamBench Curation | Historical 1962–2024 problems and computational variants with closed-form targets | M6 | ORIGINAL_REQUEST §R1 |
| 25 | 1,000 GSM8K Test Curation | 1,000 test-split problems with exact integer ground-truth extraction | M6 | ORIGINAL_REQUEST §R1 |
| 26 | 1,000 SVAMP Challenge Curation | Complete 1,000 challenge dataset with float/integer tolerances | M6 | ORIGINAL_REQUEST §R1 |
| 27 | Ground-Truth Normalization Pipeline | Enforce strict \boxed{} targets, normalized whitespace, LaTeX, and units | M6 | ORIGINAL_REQUEST §R3 |
| 28 | Offline JSONL Fixtures & CSV Catalogs | nemo_eval/datasets/fixtures/*_1000.jsonl and results/*_catalog.csv | M6 | ORIGINAL_REQUEST §R4 |
| 29 | Modular Loaders & Schema Integration | nemo_eval/datasets/ base.py, gsm8k.py, svamp.py, math.py, putnam.py, __init__.py | M7 | ORIGINAL_REQUEST §R1 |
| 30 | Methodological Sampling Rationale Report | Author comprehensive DATASET_SAMPLING_RATIONALE.md | M8 | ORIGINAL_REQUEST §R2 |
| 31 | Integration Testing & Automated Verification | Schema validation, evaluator compatibility, dry-runs, 100% pass rate | M9 | ORIGINAL_REQUEST §R5 |
| 32 | Adversarial Hardening & Forensic Audit | Challenger adversarial verification and Auditor forensic integrity checks | M10 | PROJECT_PATTERN |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Hardware Resource Telemetry & Value-Only Answer Extractor | `HardwareMonitor`, `StepEvent`/`EpisodeTrajectory` metrics, subprocess sandbox, `ValueExtractor` (LaTeX, regex, numeric) | None | DONE |
| M2 | Benchmark Dataset Ingestion & Polymorphic Evaluator | `MATHLoader` (50), `PutnamBenchLoader` (50), `LilaLoader` (350), `SympyMathEvaluator`, polymorphic checking, offline fixtures | None | DONE |
| M3 | Dual-Mode Evaluation Engine | `VanillaEngine` (0-tool CoT), `AgentLoop` dual-mode orchestration, split parity runner | M1, M2 | DONE |
| M4 | Automated Multi-Model Sweeps & Comprehensive Reporting | CLI sweep runner for 7 models, Markdown scorecards, comparison tables, energy/resource leaderboards under `results/` | M1, M2, M3 | DONE |
| M5 | 100% E2E Test Suite Pass & Adversarial Hardening | Pass all Tiers 1-4 E2E tests, execute Tier 5 adversarial hardening, gate validation | M1, M2, M3, M4 | DONE |
| M6 | Ground-Truth Normalization & 4x1,000 Offline Fixtures & CSV Catalogs | Generate `math_1000.jsonl`, `putnam_1000.jsonl`, `gsm8k_1000.jsonl`, `svamp_1000.jsonl` and 4 CSV catalogs in `results/` | None | DONE |
| M7 | Modular Loaders Architecture & Schema Integration | Update `base.py` (`svamp`), refactor `gsm8k.py`, create `svamp.py`, update `math.py`, `putnam.py`, `__init__.py`, `runner.py` | M6 | DONE |
| M8 | Methodological Sampling Rationale Report | Author `DATASET_SAMPLING_RATIONALE.md` with stratification tables, balance stats, taxonomy, closed-form vs proof | None | DONE |
| M9 | Integration Testing & Automated Verification Suite | Author `tests/unit/test_datasets/test_expanded_1000.py` and verify all tests pass | M6, M7 | DONE |
| M10 | Adversarial Coverage Hardening & Forensic Integrity Audit | Challenger adversarial stress tests and Forensic Auditor integrity verification | M6, M7, M8, M9 | IN_PROGRESS |

---

## Interface Contracts

### 1. `nemo_eval.telemetry.monitor` ↔ `nemo_eval.telemetry.tracer`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareMetrics:
    duration_ms: float = 0.0
    peak_ram_mb: float = 0.0
    gpu_vram_mb: float = 0.0
    gpu_power_watts: float = 0.0
    energy_joules: float = 0.0
    gpu_available: bool = False

class HardwareMonitor:
    def __init__(self, sample_interval_s: float = 0.05): ...
    def start(self) -> None: ...
    def sample_current(self) -> HardwareMetrics: ...
    def stop(self) -> HardwareMetrics: ...
```

### 2. `nemo_eval.telemetry.extractor` ↔ `nemo_eval.eval.engine`
```python
class ValueExtractor:
    @staticmethod
    def extract_value(raw_text: str, expected_type: Optional[str] = None) -> str:
        """Extracts strictly the target scalar / value, stripping formatting and prose."""
        ...
```

### 3. `nemo_eval.datasets.*` ↔ `nemo_eval.pipeline.runner`
```python
@dataclass
class BenchmarkTask:
    task_id: str
    dataset_name: str
    subdiscipline: str
    problem_text: str
    ground_truth: str
    eval_type: str  # 'math_symbolic' | 'float_tol' | 'exact' | 'set' | 'fraction'
    metadata: dict

class BaseDatasetLoader:
    def load(self, split: str = "test", limit: Optional[int] = None) -> list[BenchmarkTask]: ...
```

### 4. `nemo_eval.agents.vanilla` & `nemo_eval.agents.agent_loop` ↔ `nemo_eval.pipeline.runner`
```python
class BaseEvaluationEngine:
    def evaluate_task(self, task: BenchmarkTask, model: BaseLLMClient) -> EpisodeTrajectory: ...

class VanillaEngine(BaseEvaluationEngine): ...
class AgenticEngine(BaseEvaluationEngine): ...
```
