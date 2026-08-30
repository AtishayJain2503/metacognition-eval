# NeMo-AgentEval: Long-Horizon Agent Evaluation Harness

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hermetic CI/CD](https://img.shields.io/badge/environment-0%25%20network%20dependency-brightgreen.svg)]()
[![Evaluator](https://img.shields.io/badge/evaluator-SOTA%20AST%20%2B%20Regex-orange.svg)]()

**NeMo-AgentEval** is a deterministic, hermetic, and offline-first evaluation harness designed for LLM agent architectures (such as NVIDIA NeMo, ReAct, and Tool-Integrated Reasoning). It stress-tests agentic models across **multi-turn planning, tool orchestration, state-machine telemetry, and iterative error recovery**.

---

## 🌟 Key Features

* **9-State Trajectory Finite State Machine (FSM)**: Enforces strict lifecycle state transitions (`PLANNING` $\to$ `ACTION_SELECTION` $\to$ `TOOL_EXECUTION` $\to$ `OBSERVATION` $\to$ `VERIFICATION` $\to$ `SELF_CORRECTION` $\to$ `FINAL_SYNTHESIS` $\to$ `TERMINAL_SUCCESS / TERMINAL_FAILURE`).
* **SOTA Multi-Tier Mathematical Evaluator**:
  * **Balanced-Brace LaTeX Parser**: Recursively handles arbitrary nested `\boxed{...}` structures.
  * **Flexible Anchor Triggers**: Natural language extractors supporting `The final answer is $X`, `equals X`, and `is X`.
  * **Sandboxed Program-Aided Reasoning (PoT / TIR)**: Compiles and executes code blocks in an isolated subprocess with a 2-second timeout.
  * **Intermediate Tool Observation Fallback**: Rescues tasks where intermediate REPL steps succeed but final text synthesis is empty.
* **Hermetic Tool Sandboxes**: Built-in safe Python AST REPL, `:memory:` SQLite engine with CPU timeout progress handlers, and tabular CSV/Parquet analyzers.
* **Model Provider Integrations**: Native interfaces for **Ollama (local GPU)**, **Groq API**, **vLLM / OpenAI-compatible gateways**, and **NVIDIA NeMo NIM microservices**.

---

## 📊 GSM8K Benchmark Results (50 Test Episodes / Model)

Evaluated locally on consumer GPU hardware via Ollama with 0% network reliance:

| Model | Parameter Class | Raw Logged Score | Audited Synthesis Score | Audited Latent Reasoning Score | Net Gain | Primary Failure / Misclassification Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen2.5-Math-7B** | 7.25B | **86.0%** (43/50) | **86.0%** (43/50) | **86.0%** (43/50) | **0.0%** | Conformed strictly to delimiters; failures were genuine math errors. |
| **Qwen2.5-Math-1.5B** | 1.54B | 78.0% (39/50) | **80.0%** (40/50) | **80.0%** (40/50) | **+2.0%** | 1 task generated an executable Python script in synthesis that calculated `15.0`. |
| **DeepSeek-R1-7B** | 7.07B | 76.0% (38/50) | **80.0%** (40/50) | **80.0%** (40/50) | **+4.0%** | 2 tasks computed exact answers in REPL steps, but had empty synthesis strings. |
| **Llama3.2-3B** | 3.21B | 54.0% (27/50) | **68.0%** (34/50) | **68.0%** (34/50) | **+14.0%** | Outputted raw Python scripts in final responses without extracting the scalar. |
| **DeepSeek-R1-1.5B** | 1.78B | 46.0% (23/50) | **66.0%** (33/50) | **66.0%** (33/50) | **+20.0%** | Used natural phrasing without colons (`is $18` vs `Answer: 18`) followed by code fences. |
| **Phi4-mini-reasoning** | 3.84B | 48.0% (24/50) | **56.0%** (28/50) | **84.0%** (42/50) | **+36.0%** *(Latent)* | Derived mathematical proofs inside `<think>` traces, but got cut off by token limits. |
| **Qwen3-4B-Thinking** | 4.02B | 34.0% (17/50) | **44.0%** (22/50) | **44.0%** (22/50) | **+10.0%** | Calculated answers in REPL steps, but omitted final synthesis text. |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/your-username/nemo-agent-eval.git
cd nemo-agent-eval
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
pytest
```

### 3. Run Benchmark Sweeps
Create or edit your evaluation configuration (e.g., `gsm8k_config.json`):
```json
{
  "run_label": "GSM8K_Evaluation",
  "output_dir": "./results",
  "models": [
    {
      "name": "Qwen2.5-Math-7B",
      "provider": "ollama",
      "model_id": "mightykatun/qwen2.5-math:7b",
      "base_url": "http://localhost:11434/v1"
    }
  ],
  "datasets": [
    {
      "name": "gsm8k",
      "split": "test",
      "max_tasks": 50
    }
  ],
  "max_turns": 8,
  "temperature": 0.0
}
```

Execute the benchmark CLI:
```bash
python -m nemo_eval.cli run --config gsm8k_config.json
```

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                                CLI / Runner                                 │
└───────────────────────┬─────────────────────────────┬───────────────────────┘
                        │                             │
                        ▼                             ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│     Model Provider Layer      │     │      Benchmark Ingestion Layer        │
│  - Ollama Local GPU           │     │  - GSM8K (OpenAI Math)                │
│  - Groq Ultra-Fast API        │     │  - InfiAgent-DABench                  │
│  - OpenAI-Compatible Gateway  │     │  - BIRD-SQL / Spider 2.0-Lite         │
│  - Mock Replay Engine (CI/CD) │     │  - Tabular DataBench (CSV / Parquet)  │
└───────────────┬───────────────┘     └───────────────────┬───────────────────┘
                │                                         │
                ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent Execution Engine                            │
│  - Kahn's DAG Topological Planner (Plan Adherence Scoring - PAS)            │
│  - Tool Orchestrator & Automatic Parameter Bridging (Acc_tool, SPEA)        │
│  - 9-State Trajectory Finite State Machine & OTLP / JSONL Telemetry Tracers │
│  - Verification & Self-Correction Engine (SCSR, CEI, TOP penalties)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
