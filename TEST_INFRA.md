# E2E Test Infra: metacognition-eval

## Test Philosophy
- **Requirement-Driven & Opaque-Box**: Tests are derived strictly from `ORIGINAL_REQUEST.md` and user-facing specifications without coupling to internal private methods.
- **Progressive Testability**: Tier 1 tests verify fundamental contracts and work with early milestones, while Tiers 2–4 verify full integration and stress scenarios.
- **100% Hermetic Execution**: All tests execute deterministically offline using local fixtures and mocks without requiring live remote API tokens or GPU hardware.

---

## Feature Inventory & Test Coverage

| # | Feature | Requirement Source | Tier 1 (Feature) | Tier 2 (BVA/Corner) | Tier 3 (Pairwise) | Tier 4 (Real-World) |
|---|---------|-------------------|:----------------:|:-------------------:|:-----------------:|:-------------------:|
| 1 | HardwareTelemetryMonitor (RAM, GPU, Joules) | ORIGINAL_REQUEST §R1 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 2 | Value-Only Answer Extractor | ORIGINAL_REQUEST §R1 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 3 | Sandboxed Subprocess Execution | ORIGINAL_REQUEST §AC | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 4 | MATH Dataset Loader (50 samples, LaTeX) | ORIGINAL_REQUEST §R3 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 5 | PutnamBench Loader (50 competition tasks) | ORIGINAL_REQUEST §R3 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 6 | Lila Loader (7 categories x 50 = 350 tasks) | ORIGINAL_REQUEST §R3 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 7 | Polymorphic & SymPy Ground Truth Checking | ORIGINAL_REQUEST §R3 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 8 | Vanilla Zero-Shot Engine | ORIGINAL_REQUEST §R2 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 9 | 9-State Agentic FSM Engine & REPL | ORIGINAL_REQUEST §R2 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 10 | Dual-Mode Split Parity & Delta Metrics | ORIGINAL_REQUEST §R2 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 11 | 7 Target Model Configurations | ORIGINAL_REQUEST §R4 | ≥5 tests | ≥5 tests | ✓ | ✓ |
| 12 | CLI Sweeps & Markdown/JSONL Reporting | ORIGINAL_REQUEST §R4 | ≥5 tests | ≥5 tests | ✓ | ✓ |

---

## Test Architecture

### 1. Test Runner & Invocation
- Framework: `pytest`
- Invocation: `pytest tests/e2e/test_metacognition_eval_e2e.py -v`
- Pass/Fail Semantics: 100% pass, exit code 0.

### 2. Test File Layout
- `tests/e2e/test_metacognition_eval_e2e.py`: Master E2E test suite covering Tiers 1–4.
- `tests/unit/test_telemetry/`: Unit tests for `HardwareMonitor`, `StepEvent`, `ValueExtractor`.
- `tests/unit/test_datasets/`: Unit tests for `MATHLoader`, `PutnamBenchLoader`, `LilaLoader`.
- `tests/unit/test_eval/`: Unit tests for `SympyMathEvaluator` and polymorphic matchers.
- `tests/unit/test_pipeline/`: Unit tests for `VanillaEngine`, `AgenticEngine`, and CLI sweeps.

---

## Test Tier Definitions

### Tier 1 — Feature Coverage (Isolated Happy Path)
- Verify `HardwareMonitor.start()` and `.stop()` produces non-zero `duration_ms` and valid `peak_ram_mb`.
- Verify `ValueExtractor.extract_value()` handles standard `\boxed{42}`, `\boxed{\frac{1}{2}}`, raw numbers, and strings.
- Verify `MATHLoader.load(split='test', limit=50)` returns exactly 50 tasks with valid problem text and LaTeX boxed ground truth.
- Verify `PutnamBenchLoader.load(split='test', limit=50)` returns 50 competition tasks with formal expressions.
- Verify `LilaLoader.load(split='test', limit=50)` returns 50 tasks for each of the 7 subcategories (350 total).
- Verify `SympyMathEvaluator.evaluate()` recognizes mathematical equivalence (e.g. `2*x + 4 == 2*(x+2)`).
- Verify `VanillaEngine.evaluate_task()` executes zero-shot without invoking tool calls.
- Verify `AgenticEngine.evaluate_task()` runs multi-turn FSM with REPL tool execution.
- Verify `BenchmarkRunner` accepts all 7 model identifiers without errors.
- Verify `MarkdownReporter` formats scorecards and comparison tables.

### Tier 2 — Boundary & Corner Cases (Error Handling & Limits)
- Telemetry on environments with no GPU (verify graceful fallback: `gpu_vram_mb == 0.0`, `energy_joules >= 0.0`, no crashes).
- ValueExtractor with deeply nested braces `\boxed{\frac{\sqrt{x^2+1}}{2}}`, multiple boxed expressions, empty strings, malformed LaTeX, markdown code blocks, prose-only outputs.
- Subprocess sandbox with infinite loops (`while True: pass`), memory bombs (`'x' * 10**9`), syntax errors, prohibited OS operations (AST security validation).
- Dataset loaders with limit=0, limit=1, limit=50, limit=1000 (clamped), missing fixture fallback, invalid split name.
- SymPy evaluator with complex expressions, division by zero, non-algebraic strings, complex numbers, sets (`{1, 2}` vs `{2, 1}`), fractions (`3/6` vs `1/2`).
- Dual-mode runner handling model timeouts, empty responses, and malformed completions gracefully.

### Tier 3 — Cross-Feature Combinations (Pairwise Coverage)
- `MATHLoader` tasks evaluated with `VanillaEngine` + `ValueExtractor` + `SympyMathEvaluator` + `HardwareTelemetry`.
- `PutnamBenchLoader` tasks evaluated with `AgenticEngine` + `PythonREPL` + `SelfCorrection` + `HardwareTelemetry`.
- `LilaLoader` tasks evaluated across both `Vanilla` and `Agentic` modes to compute exact delta performance tables.
- Model sweep runner processing multiple datasets sequentially while maintaining streaming JSONL telemetry integrity and memory isolation.

### Tier 4 — Real-World Application Scenarios
- End-to-end full evaluation sweep over 7 models across sample slices of MATH, Putnam, and Lila datasets in both Vanilla and Agentic modes, generating complete summary scorecards and leaderboards under `results/`.
- Simulation of long-running multi-turn agentic reasoning session with iterative Python REPL debugging, step-level energy accounting, and final answer extraction.

---

## Coverage Thresholds
- Tier 1: ≥5 test cases per feature
- Tier 2: ≥5 test cases per feature
- Tier 3: Pairwise coverage across loaders, engines, evaluators, and telemetry
- Tier 4: ≥5 realistic full-pipeline application scenarios
- **Total E2E test cases: ≥70 comprehensive tests**
