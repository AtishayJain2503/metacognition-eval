# GSM8K_SLM_Benchmark_2026 — Summary Report

> All metrics are offline-hermetic (0% network dependency).

## Aggregate Results by Model × Dataset

| Model | Dataset | Tasks | Success% | GT Score | PAS | Acc_tool | SPEA | SCSR | CEI | TOP |
|-------|---------|-------|----------|----------|-----|----------|------|------|-----|-----|
| DeepSeek-R1-1.5B | gsm8k | 50 | 100.0% | 0.4600 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0544 |
| Qwen2.5-Math-1.5B | gsm8k | 50 | 100.0% | 0.7800 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0728 |
| Phi4-mini-reasoning | gsm8k | 50 | 100.0% | 0.4800 | 1.0000 | 0.9667 | 1.0000 | 0.9500 | 0.9500 | 0.0744 |
| Qwen3-4B-Thinking | gsm8k | 50 | 100.0% | 0.3400 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0048 |
| Llama3.2-3B | gsm8k | 50 | 100.0% | 0.5400 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0336 |
| DeepSeek-R1-7B | gsm8k | 50 | 100.0% | 0.7600 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0656 |
| Qwen2.5-Math-7B | gsm8k | 50 | 92.0% | 0.8600 | 1.0000 | 1.0000 | 1.0000 | 0.9800 | 0.9800 | 0.0736 |

## Metric Definitions

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **GT Score** | Ground truth eval engine | Final answer correctness |
| **PAS** | LCS(planned_order, actual_order) / len(plan) | Plan adherence during execution |
| **Acc_tool** | valid_tool_calls / total_calls | Tool selection correctness |
| **SPEA** | bridged_successes / bridged_calls | Parameter bridging quality |
| **SCSR** | successful_recoveries / attempts | Self-correction success rate |
| **CEI** | recoveries / correction_turns | Correction efficiency |
| **TOP** | correction_turns / max_turns | Turn overhead penalty |

- [DeepSeek-R1-1.5B × gsm8k](./scorecard_DeepSeek-R1-1.5B_gsm8k.md)
- [Qwen2.5-Math-1.5B × gsm8k](./scorecard_Qwen2.5-Math-1.5B_gsm8k.md)
- [Phi4-mini-reasoning × gsm8k](./scorecard_Phi4-mini-reasoning_gsm8k.md)
- [Qwen3-4B-Thinking × gsm8k](./scorecard_Qwen3-4B-Thinking_gsm8k.md)
- [Llama3.2-3B × gsm8k](./scorecard_Llama3.2-3B_gsm8k.md)
- [DeepSeek-R1-7B × gsm8k](./scorecard_DeepSeek-R1-7B_gsm8k.md)
- [Qwen2.5-Math-7B × gsm8k](./scorecard_Qwen2.5-Math-7B_gsm8k.md)