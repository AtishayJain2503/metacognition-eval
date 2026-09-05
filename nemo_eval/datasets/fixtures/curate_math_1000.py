"""
nemo_eval.datasets.fixtures.curate_math_1000
============================================
Curate and generate the 1,000-sample Hendrycks MATH benchmark suite.

- Ingests 5,000 test tasks from EleutherAI/hendrycks_math across 7 subjects and 5 levels.
- Applies mathematically sound stratified sampling (20% proportional sampling, seed=42)
  preserving proportional representation across subjects and difficulty levels.
- Normalizes ground truths to clean LaTeX \\boxed{...}.
- Exports offline fixture to nemo_eval/datasets/fixtures/math_1000.jsonl.
- Exports comprehensive catalog to results/math_tasks_catalog.csv.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from nemo_eval.datasets.base import BenchmarkTask
from nemo_eval.eval.math_eval import extract_latex_boxed


CONFIG_TO_SUBJECT = {
    "algebra": "Algebra",
    "counting_and_probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate_algebra": "Intermediate Algebra",
    "number_theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}

SUBJECT_SLUGS = {
    "algebra": "algebra",
    "counting_and_probability": "counting_and_prob",
    "geometry": "geometry",
    "intermediate_algebra": "intermediate_alg",
    "number_theory": "number_theory",
    "prealgebra": "prealgebra",
    "precalculus": "precalculus",
}

QUERY_TEMPLATE = "{problem}\n\nSolve this mathematical problem step-by-step. Put your final answer within \\boxed{{}}."


def load_all_hendrycks_math_test() -> List[Dict[str, Any]]:
    """Load all 5,000 test tasks across all 7 subject disciplines."""
    raw_tasks: List[Dict[str, Any]] = []
    for config, subject_name in CONFIG_TO_SUBJECT.items():
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for orig_idx, item in enumerate(ds):
            lvl_str = item["level"].strip()
            level = int(lvl_str.replace("Level ", "")) if "Level" in lvl_str else int(lvl_str)
            raw_tasks.append({
                "config": config,
                "subject": subject_name,
                "level": level,
                "orig_idx": orig_idx,
                "problem": item["problem"].strip(),
                "solution": item["solution"].strip(),
            })
    return raw_tasks


def compute_stratified_sample_quotas(
    tasks: List[Dict[str, Any]], target_total: int = 1000
) -> Dict[Tuple[str, int], int]:
    """
    Compute mathematically sound proportional quotas using the Hamilton-Hare
    (Largest Remainder) method in two stages:
    1. Allocate target_total proportionally across subjects.
    2. Within each subject, allocate its quota proportionally across difficulty levels 1-5.
    """
    # Count population per stratum
    subject_counts: Dict[str, int] = defaultdict(int)
    stratum_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for t in tasks:
        subject_counts[t["config"]] += 1
        stratum_counts[(t["config"], t["level"])] += 1

    total_pop = len(tasks)
    sampling_fraction = target_total / total_pop  # 1000 / 5000 = 0.20

    # Stage 1: Subject quotas
    subject_quotas: Dict[str, int] = {}
    subj_exact = {c: count * sampling_fraction for c, count in subject_counts.items()}
    subj_alloc = {c: int(math.floor(q)) for c, q in subj_exact.items()}
    subj_rem = sorted([(subj_exact[c] - subj_alloc[c], c) for c in subject_counts], reverse=True)
    needed_subj = target_total - sum(subj_alloc.values())
    for i in range(needed_subj):
        subj_alloc[subj_rem[i][1]] += 1
    subject_quotas = subj_alloc

    # Stage 2: Apportion within each subject across levels 1-5
    stratum_quotas: Dict[Tuple[str, int], int] = {}
    for c in CONFIG_TO_SUBJECT:
        n_c = subject_quotas[c]
        total_c = subject_counts[c]
        lvl_exact = {lvl: n_c * (stratum_counts[(c, lvl)] / total_c) for lvl in range(1, 6)}
        lvl_alloc = {lvl: int(math.floor(lvl_exact[lvl])) for lvl in range(1, 6)}
        lvl_rem = sorted(
            [(lvl_exact[lvl] - lvl_alloc[lvl], lvl) for lvl in range(1, 6)],
            reverse=True,
        )
        needed_lvl = n_c - sum(lvl_alloc.values())
        for i in range(needed_lvl):
            lvl_alloc[lvl_rem[i][1]] += 1
        for lvl in range(1, 6):
            stratum_quotas[(c, lvl)] = lvl_alloc[lvl]

    return stratum_quotas


def sample_tasks(
    tasks: List[Dict[str, Any]], quotas: Dict[Tuple[str, int], int], seed: int = 42
) -> List[Dict[str, Any]]:
    """Sample tasks deterministically according to computed quotas."""
    # Group tasks by stratum
    strata: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        strata[(t["config"], t["level"])].append(t)

    # Sort each stratum deterministically by orig_idx
    for key in strata:
        strata[key].sort(key=lambda x: x["orig_idx"])

    # Sample using seeded RNG
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []

    # Iterate over sorted keys for deterministic execution order
    for key in sorted(strata.keys()):
        count = quotas[key]
        candidates = strata[key]
        chosen_indices = sorted(rng.sample(range(len(candidates)), count))
        for idx in chosen_indices:
            selected.append(candidates[idx])

    return selected


def curate_and_export() -> Tuple[Path, Path]:
    """Curate 1,000 MATH tasks and write JSONL fixture and CSV catalog."""
    print("Ingesting 5,000 test tasks from EleutherAI/hendrycks_math...")
    raw_tasks = load_all_hendrycks_math_test()
    assert len(raw_tasks) == 5000, f"Expected 5,000 tasks, got {len(raw_tasks)}"

    print("Computing proportional stratified quotas across 7 subjects and 5 levels...")
    quotas = compute_stratified_sample_quotas(raw_tasks, target_total=1000)
    assert sum(quotas.values()) == 1000, f"Expected 1,000 quota sum, got {sum(quotas.values())}"

    print("Sampling 1,000 tasks with seed=42...")
    sampled_raw = sample_tasks(raw_tasks, quotas, seed=42)
    assert len(sampled_raw) == 1000, f"Expected 1,000 sampled tasks, got {len(sampled_raw)}"

    # Standardize tasks
    curated_records: List[Dict[str, Any]] = []
    catalog_rows: List[Dict[str, Any]] = []

    for i, t in enumerate(sampled_raw):
        task_idx = i + 1
        subject_name = t["subject"]
        config = t["config"]
        level = t["level"]
        subject_slug = SUBJECT_SLUGS[config]

        # Extract and normalize ground truth
        extracted_inner = extract_latex_boxed(t["solution"])
        if not extracted_inner:
            raise ValueError(f"Failed to extract \\boxed{{}} from solution at index {t['orig_idx']}")
        clean_inner = extracted_inner.strip()
        ground_truth = f"\\boxed{{{clean_inner}}}"

        task_id = f"math_{subject_slug}_lvl{level}_{task_idx:04d}"
        query = QUERY_TEMPLATE.format(problem=t["problem"])

        record = {
            "task_id": task_id,
            "benchmark_name": "math",
            "query": query,
            "ground_truth": ground_truth,
            "eval_type": "math_symbolic",
            "metadata": {
                "subject": subject_name,
                "level": level,
                "split": "test",
                "boxed_solution": clean_inner,
                "source": "EleutherAI/hendrycks_math",
                "original_config": config,
                "original_index": t["orig_idx"],
            },
        }

        # Verify BenchmarkTask schema validation
        task_obj = BenchmarkTask.model_validate(record)
        assert task_obj.task_id == task_id
        assert task_obj.benchmark_name == "math"
        assert task_obj.eval_type == "math_symbolic"

        curated_records.append(record)

        catalog_rows.append({
            "task_index": task_idx,
            "task_id": task_id,
            "category": subject_name,
            "level": level,
            "eval_type": "math_symbolic",
            "ground_truth": ground_truth,
            "query": query,
        })

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    fixtures_dir = project_root / "nemo_eval" / "datasets" / "fixtures"
    results_dir = project_root / "results"

    fixtures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = fixtures_dir / "math_1000.jsonl"
    csv_path = results_dir / "math_tasks_catalog.csv"

    print(f"Writing JSONL fixture to {jsonl_path}...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in curated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Writing CSV catalog to {csv_path}...")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["task_index", "task_id", "category", "level", "eval_type", "ground_truth", "query"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog_rows)

    print(f"Successfully generated {len(curated_records)} tasks!")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
    return jsonl_path, csv_path


if __name__ == "__main__":
    curate_and_export()
