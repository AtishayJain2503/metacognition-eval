"""
nemo_eval.datasets.gsm8k
=========================
GSM8K (Grade School Math 8K) dataset loader.

Downloads and parses 50 samples from the openai/gsm8k dataset via HuggingFace.
Ground truth is the integer answer extracted from the '#### <answer>' annotation.
Evaluation type: float_tol (exact integer match with zero tolerance).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nemo_eval.datasets.base import BenchmarkTask


def _extract_answer(solution_text: str) -> Optional[int]:
    """Extract the integer answer after '####' in GSM8K solution strings."""
    match = re.search(r"####\s*([\-0-9,]+)", solution_text)
    if match:
        raw = match.group(1).replace(",", "").strip()
        try:
            return int(raw)
        except ValueError:
            return None
    return None


class GSM8KLoader:
    """
    Loads GSM8K tasks from HuggingFace datasets library.

    Each task is converted to a BenchmarkTask where:
    - query: the word problem as a natural language question
    - ground_truth: the integer answer (extracted from '#### N' annotation)
    - eval_type: 'float_tol' (allows tiny floating point variance)
    - db_path / table_path: None (pure math, no external tools needed)
    """

    HF_DATASET_ID = "openai/gsm8k"
    HF_CONFIG = "main"

    def __init__(self, split: str = "test", max_tasks: Optional[int] = 50):
        self.split = split
        self.max_tasks = max_tasks

    def load(self, split: Optional[str] = None) -> List[BenchmarkTask]:
        """Download and parse GSM8K samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for GSM8K. "
                "Install it with: pip install datasets"
            )

        active_split = split or self.split
        print(f"[GSM8KLoader] Downloading {self.HF_DATASET_ID} ({active_split} split)...")
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=active_split, trust_remote_code=False)

        tasks: List[BenchmarkTask] = []
        count = 0

        for i, row in enumerate(ds):
            if self.max_tasks and count >= self.max_tasks:
                break

            question: str = row["question"].strip()
            answer_text: str = row["answer"].strip()
            gt_int = _extract_answer(answer_text)

            if gt_int is None:
                # Skip malformed entries
                continue

            task_id = f"gsm8k_{active_split}_{i:04d}"

            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    benchmark_name="synthetic",   # reuse 'synthetic' literal — avoids schema changes
                    query=(
                        f"{question}\n\n"
                        "Solve this step-by-step using Python code. "
                        "Your final answer must be an integer."
                    ),
                    ground_truth=float(gt_int),   # float for float_tol evaluator
                    eval_type="float_tol",
                    db_path=None,
                    table_path=None,
                    metadata={
                        "source": "gsm8k",
                        "original_answer": answer_text,
                        "split": active_split,
                        "index": i,
                        "tolerance": 0.5,   # accept ±0.5 to handle float<->int rounding
                    },
                )
            )
            count += 1

        print(f"[GSM8KLoader] Loaded {len(tasks)} tasks from GSM8K.")
        return tasks
