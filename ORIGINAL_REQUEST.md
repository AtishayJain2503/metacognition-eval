# Original User Request

## Initial Request — 2026-09-05T08:30:54Z

Use a very large team of agents.

Expand and curate benchmark suites for Hendrycks MATH, PutnamBench, GSM8K, and SVAMP to 1,000 samples each with mathematically sound sampling rationales, normalized closed-form ground truths, offline .jsonl fixtures, and CSV catalogs.

Working directory: c:/Projects/MetaCognition
Integrity mode: development

## Requirements

### R1. 1,000-Sample Ingestion & Loader Architecture
Expand or implement modular dataset loaders in nemo_eval/datasets/ for:
1. Hendrycks MATH: Ingest 1,000 samples selected via stratified sampling across all 7 subject disciplines (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus) and difficulty Levels 1–5.
2. PutnamBench: Ingest 1,000 samples leveraging the extended historical and computational competition dataset (years 1962–2024 / Trishul Lab formalization suite and curated computational variants), ensuring deterministic closed-form targets.
3. GSM8K: Ingest 1,000 test-split grade-school math word problems from OpenAI GSM8K with exact integer ground-truth extraction.
4. SVAMP: Ingest the complete 1,000-sample SVAMP challenge dataset (Patel et al. / Ritvik word problem variations) with float/integer tolerances.

### R2. Methodological Sampling Rationale Report
Author a comprehensive documentation artifact (DATASET_SAMPLING_RATIONALE.md) providing a rigorous scientific rationale for the subset selection:
- Explain the stratification distribution across mathematical difficulty and domain taxonomy.
- Contrast closed-form computational evaluation vs. open-ended theorem proving.
- Provide balance statistics showing proportional representation across difficulty tiers.

### R3. Ground-Truth Normalization & Cleaning Pipeline
Standardize all 4,000 tasks following the clean Putnam pipeline:
- Extract and verify deterministic numerical, fractional, or symbolic targets.
- Enforce strict \boxed{} representation in ground truth solutions.
- Normalize whitespace, LaTeX syntax, and units to prevent parsing artifacts.

### R4. Offline Fixture Generation & CSV Catalog Exports
Generate reproducible offline assets in nemo_eval/datasets/fixtures/ and results/:
- Save clean .jsonl fixture files (math_1000.jsonl, putnam_1000.jsonl, gsm8k_1000.jsonl, svamp_1000.jsonl).
- Export comprehensive CSV catalogs with task index, ID, category/subject, difficulty level, query text, and ground truth.

### R5. Integration Testing & Automated Verification
Integrate the 4 expanded loaders into nemo_eval/datasets/__init__.py and create unit tests verifying:
- Schema conformance for all 1,000 records per dataset.
- Evaluator compatibility across math_symbolic, float_tol, fraction, and exact.
- 100% test pass rate across the existing 967 tests and newly added dataset tests.

## Acceptance Criteria

### Dataset Scale & Integrity
- [ ] Exactly 1,000 valid, non-corrupted benchmark tasks exist for Hendrycks MATH, PutnamBench, GSM8K, and SVAMP.
- [ ] Each task contains non-empty task_id, query, ground_truth, eval_type, and domain metadata.
- [ ] All ground-truth answers are normalized and enclosed in \boxed{} or numerical format.

### Scientific Rationale & Documentation
- [ ] DATASET_SAMPLING_RATIONALE.md contains detailed methodology, domain coverage tables, and sampling justifications suitable for a mentor review or research paper appendix.

### Exports & Reproducibility
- [ ] 4 fixture .jsonl files and 4 .csv catalogs generated in the repository.
- [ ] Dry-run evaluation tests confirm zero schema errors and non-zero accuracy on known sample solutions.
- [ ] All pytest test suites pass without regression.

## Follow-up Directive — 2026-09-05T08:48:22Z

User directive update:
1. Thoroughly validate each sample in the 1,000-sample suites (ensure zero corrupted lines, valid \boxed{} ground truth, valid evaluation types, non-empty questions).
2. Verify code logic and run full test suites before beginning any benchmark execution.
3. Prepare for high-throughput benchmarking: configure multi-worker/parallel execution where possible, and leverage checkpoint resumption for identical unbiased prior runs if valid.
4. Keep reporting progress every 30 minutes.
