"""
tests.unit.test_datasets.test_expanded_1000
============================================
Comprehensive unit and integration test suite for the expanded 1,000-sample
benchmark suites across Hendrycks MATH, PutnamBench, GSM8K, and SVAMP (4,000 tasks total).

Validates:
1. Exact Task Counts & Fixture Presence (1,000 per benchmark, 4,000 total).
2. CSV Catalog Integrity (results/*_catalog.csv, 1,001 rows, required columns).
3. Schema Conformance for all 4,000 tasks (BenchmarkTask schema, uniqueness, non-empty fields).
4. Ground Truth Normalization (enclosed in \\boxed{...}, brace balance, clean whitespace).
5. Evaluator Compatibility & Self-Consistency across all 4 eval types (math_symbolic, float_tol, fraction, exact).
6. Dry-Run Evaluation on Sample Tasks (non-zero accuracy, CoT isolation, zero schema errors).
7. Stratification Integrity (7 MATH subjects & levels 1-5, 7 Putnam subdisciplines 1962-2024, GSM8K integers, SVAMP 4 operations).
8. Backward Compatibility & Legacy Invariants (default 50 tasks, 50+50+350=450 invariant).
9. Adversarial & Boundary Robustness (empty inputs, malformed candidates, nonexistent IDs).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.gsm8k import GSM8KLoader
from nemo_eval.datasets.lila import LilaLoader
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.datasets.putnam import PutnamBenchLoader
from nemo_eval.datasets.svamp import SVAMPLoader
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.engine import evaluate_task_result
from nemo_eval.eval.math_eval import extract_latex_boxed


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "nemo_eval" / "datasets" / "fixtures"
RESULTS_DIR = REPO_ROOT / "results"

MATH_JSONL = FIXTURES_DIR / "math_1000.jsonl"
PUTNAM_JSONL = FIXTURES_DIR / "putnam_1000.jsonl"
GSM8K_JSONL = FIXTURES_DIR / "gsm8k_1000.jsonl"
SVAMP_JSONL = FIXTURES_DIR / "svamp_1000.jsonl"

MATH_CSV = RESULTS_DIR / "math_tasks_catalog.csv"
PUTNAM_CSV = RESULTS_DIR / "putnam_tasks_catalog.csv"
GSM8K_CSV = RESULTS_DIR / "gsm8k_tasks_catalog.csv"
SVAMP_CSV = RESULTS_DIR / "svamp_tasks_catalog.csv"

REQUIRED_CSV_COLUMNS = [
    "task_index",
    "task_id",
    "category",
    "level",
    "eval_type",
    "ground_truth",
    "query",
]

ALLOWED_EVAL_TYPES = {"math_symbolic", "float_tol", "fraction", "exact"}


# =============================================================================
# Shared Module-Level Fixtures (Cached for high-speed execution across 4,000 tasks)
# =============================================================================

@pytest.fixture(scope="module")
def math_tasks_1000() -> List[BenchmarkTask]:
    """Load the full 1,000 Hendrycks MATH benchmark tasks."""
    loader = MATHLoader(use_1000=True)
    tasks = loader.load_tasks()
    assert len(tasks) == 1000, f"Expected 1,000 MATH tasks, got {len(tasks)}"
    return tasks


@pytest.fixture(scope="module")
def putnam_tasks_1000() -> List[BenchmarkTask]:
    """Load the full 1,000 PutnamBench competition tasks."""
    loader = PutnamBenchLoader(use_1000=True)
    tasks = loader.load_tasks()
    assert len(tasks) == 1000, f"Expected 1,000 Putnam tasks, got {len(tasks)}"
    return tasks


@pytest.fixture(scope="module")
def gsm8k_tasks_1000() -> List[BenchmarkTask]:
    """Load the full 1,000 GSM8K reasoning tasks."""
    loader = GSM8KLoader()
    tasks = loader.load_tasks()
    assert len(tasks) == 1000, f"Expected 1,000 GSM8K tasks, got {len(tasks)}"
    return tasks


@pytest.fixture(scope="module")
def svamp_tasks_1000() -> List[BenchmarkTask]:
    """Load the full 1,000 SVAMP challenge tasks."""
    loader = SVAMPLoader()
    tasks = loader.load_tasks()
    assert len(tasks) == 1000, f"Expected 1,000 SVAMP tasks, got {len(tasks)}"
    return tasks


@pytest.fixture(scope="module")
def all_4000_tasks(
    math_tasks_1000: List[BenchmarkTask],
    putnam_tasks_1000: List[BenchmarkTask],
    gsm8k_tasks_1000: List[BenchmarkTask],
    svamp_tasks_1000: List[BenchmarkTask],
) -> List[BenchmarkTask]:
    """Aggregate all 4,000 benchmark tasks across the four suites."""
    combined = math_tasks_1000 + putnam_tasks_1000 + gsm8k_tasks_1000 + svamp_tasks_1000
    assert len(combined) == 4000, f"Expected 4,000 combined tasks, got {len(combined)}"
    return combined


# =============================================================================
# 1. Exact Task Counts & Fixture Presence
# =============================================================================

class TestExactTaskCountsAndFixtures:
    """Verifies existence of all offline .jsonl fixtures and exact 1,000-count ingestion."""

    def test_all_four_fixture_files_exist(self):
        """Verify that all four 1,000-sample offline JSONL fixtures exist on disk."""
        assert MATH_JSONL.is_file(), f"Missing fixture: {MATH_JSONL}"
        assert PUTNAM_JSONL.is_file(), f"Missing fixture: {PUTNAM_JSONL}"
        assert GSM8K_JSONL.is_file(), f"Missing fixture: {GSM8K_JSONL}"
        assert SVAMP_JSONL.is_file(), f"Missing fixture: {SVAMP_JSONL}"

    def test_fixture_files_contain_exact_1000_json_lines(self):
        """Verify that every JSONL fixture contains exactly 1,000 non-empty JSON lines."""
        for path in (MATH_JSONL, PUTNAM_JSONL, GSM8K_JSONL, SVAMP_JSONL):
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1000, f"Fixture {path.name} has {len(lines)} lines, expected 1,000"

    def test_math_loader_exact_1000_tasks(self, math_tasks_1000: List[BenchmarkTask]):
        """Verify MATHLoader loads exactly 1,000 tasks when use_1000=True."""
        assert len(math_tasks_1000) == 1000
        # Verify via dynamic fixture_name as well
        custom_loader = MATHLoader(fixture_name="math_1000.jsonl", max_tasks=1000)
        assert len(custom_loader.load_tasks()) == 1000
        # Verify via load(use_1000=True)
        assert len(MATHLoader().load(use_1000=True)) == 1000

    def test_putnam_loader_exact_1000_tasks(self, putnam_tasks_1000: List[BenchmarkTask]):
        """Verify PutnamBenchLoader loads exactly 1,000 tasks when use_1000=True."""
        assert len(putnam_tasks_1000) == 1000
        # Verify via dynamic fixture_name
        custom_loader = PutnamBenchLoader(fixture_name="putnam_1000.jsonl", max_tasks=1000)
        assert len(custom_loader.load_tasks()) == 1000
        # Verify via load(use_1000=True)
        assert len(PutnamBenchLoader().load(use_1000=True)) == 1000

    def test_gsm8k_loader_exact_1000_tasks(self, gsm8k_tasks_1000: List[BenchmarkTask]):
        """Verify GSM8KLoader loads exactly 1,000 tasks by default."""
        assert len(gsm8k_tasks_1000) == 1000
        # Verify via load() method
        assert len(GSM8KLoader().load()) == 1000

    def test_svamp_loader_exact_1000_tasks(self, svamp_tasks_1000: List[BenchmarkTask]):
        """Verify SVAMPLoader loads exactly 1,000 tasks by default."""
        assert len(svamp_tasks_1000) == 1000
        # Verify via load() method
        assert len(SVAMPLoader().load()) == 1000

    def test_total_expanded_suite_exact_4000_tasks(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify the aggregate expanded benchmark suite contains exactly 4,000 tasks."""
        assert len(all_4000_tasks) == 4000

    def test_loader_limit_subsets(self):
        """Verify that all four loaders honor limit parameters on expanded suites."""
        math_10 = MATHLoader(use_1000=True).load_tasks(limit=10)
        assert len(math_10) == 10

        putnam_25 = PutnamBenchLoader(use_1000=True).load_tasks(limit=25)
        assert len(putnam_25) == 25

        gsm8k_5 = GSM8KLoader().load(limit=5)
        assert len(gsm8k_5) == 5

        svamp_15 = SVAMPLoader().load(limit=15)
        assert len(svamp_15) == 15


# =============================================================================
# 2. CSV Catalog Integrity
# =============================================================================

class TestCsvCatalogIntegrity:
    """Verifies structure, row counts, and content alignment of generated CSV catalogs."""

    @pytest.mark.parametrize(
        "csv_path,benchmark_name",
        [
            (MATH_CSV, "math"),
            (PUTNAM_CSV, "putnam"),
            (GSM8K_CSV, "gsm8k"),
            (SVAMP_CSV, "svamp"),
        ],
    )
    def test_csv_catalogs_exist(self, csv_path: Path, benchmark_name: str):
        """Verify that catalog file exists in results directory."""
        assert csv_path.is_file(), f"Missing catalog CSV: {csv_path}"

    @pytest.mark.parametrize(
        "csv_path,benchmark_name",
        [
            (MATH_CSV, "math"),
            (PUTNAM_CSV, "putnam"),
            (GSM8K_CSV, "gsm8k"),
            (SVAMP_CSV, "svamp"),
        ],
    )
    def test_csv_catalogs_exact_1001_rows(self, csv_path: Path, benchmark_name: str):
        """Verify that each CSV catalog has exactly 1,001 rows (1 header + 1,000 data rows)."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 1001, f"{csv_path.name} has {len(rows)} rows; expected 1001"

    @pytest.mark.parametrize(
        "csv_path,benchmark_name",
        [
            (MATH_CSV, "math"),
            (PUTNAM_CSV, "putnam"),
            (GSM8K_CSV, "gsm8k"),
            (SVAMP_CSV, "svamp"),
        ],
    )
    def test_csv_catalogs_required_headers(self, csv_path: Path, benchmark_name: str):
        """Verify that each CSV catalog contains the required columns."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == REQUIRED_CSV_COLUMNS, f"Header mismatch in {csv_path.name}: {header}"

    def test_csv_catalogs_alignment_with_benchmark_tasks(
        self,
        math_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
    ):
        """Verify that catalog entries match loaded BenchmarkTask records in ID, eval_type, and ground_truth."""
        datasets = [
            (MATH_CSV, math_tasks_1000),
            (PUTNAM_CSV, putnam_tasks_1000),
            (GSM8K_CSV, gsm8k_tasks_1000),
            (SVAMP_CSV, svamp_tasks_1000),
        ]
        for csv_path, tasks in datasets:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == len(tasks) == 1000
            for idx, (row, task) in enumerate(zip(rows, tasks), start=1):
                assert int(row["task_index"]) == idx
                assert row["task_id"] == task.task_id
                assert row["eval_type"] == task.eval_type
                assert row["ground_truth"] == task.ground_truth
                # Query in CSV may have normalized spaces/newlines to maintain clean tabular formatting
                assert " ".join(row["query"].split()) == " ".join(task.query.split())


# =============================================================================
# 3. Schema Conformance for all 4,000 tasks
# =============================================================================

class TestSchemaConformance4000:
    """Validates full Pydantic schema compliance, uniqueness, and fields across all 4,000 tasks."""

    def test_all_tasks_are_valid_benchmark_task_instances(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify all 4,000 tasks are valid BenchmarkTask instances."""
        for t in all_4000_tasks:
            assert isinstance(t, BenchmarkTask)

    def test_global_task_id_uniqueness_across_all_4000_tasks(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify that all 4,000 tasks have globally unique task_ids across the entire benchmark suite."""
        all_ids = [t.task_id for t in all_4000_tasks]
        assert len(all_ids) == 4000
        unique_ids = set(all_ids)
        assert len(unique_ids) == 4000, f"Found {4000 - len(unique_ids)} duplicate task IDs!"

    def test_task_id_conventions_and_prefixes(
        self,
        math_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
    ):
        """Verify task ID formatting and benchmark-specific prefixes."""
        for t in math_tasks_1000:
            assert t.task_id.startswith("math_")
        for t in putnam_tasks_1000:
            assert t.task_id.startswith("putnam_")
        for t in gsm8k_tasks_1000:
            assert t.task_id.startswith("gsm8k_")
        for t in svamp_tasks_1000:
            assert t.task_id.startswith("svamp_")

    def test_query_non_empty_and_meaningful(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify every task has a non-empty, meaningful query string."""
        for t in all_4000_tasks:
            assert isinstance(t.query, str)
            assert len(t.query.strip()) >= 5, f"Query too short for task {t.task_id}"

    def test_ground_truth_non_empty(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify every task has a non-empty ground truth value."""
        for t in all_4000_tasks:
            assert t.ground_truth is not None
            assert len(str(t.ground_truth).strip()) > 0, f"Empty ground truth for task {t.task_id}"

    def test_eval_type_in_valid_set(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify every task has a valid eval_type in the recognized polymorphic set."""
        for t in all_4000_tasks:
            assert t.eval_type in ALLOWED_EVAL_TYPES, f"Invalid eval_type {t.eval_type} for task {t.task_id}"

    def test_benchmark_name_matches_suite(
        self,
        math_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
    ):
        """Verify benchmark_name matches the respective suite."""
        for t in math_tasks_1000:
            assert t.benchmark_name == "math"
        for t in putnam_tasks_1000:
            assert t.benchmark_name == "putnam"
        for t in gsm8k_tasks_1000:
            assert t.benchmark_name == "gsm8k"
        for t in svamp_tasks_1000:
            assert t.benchmark_name == "svamp"

    def test_metadata_completeness(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify every task contains rich metadata with domain/category information."""
        for t in all_4000_tasks:
            assert isinstance(t.metadata, dict)
            assert len(t.metadata) > 0, f"Empty metadata for task {t.task_id}"
            # All tasks must have a category or subject or type
            has_domain = any(k in t.metadata for k in ("category", "subject", "subdiscipline", "type"))
            assert has_domain, f"No domain categorization in metadata for {t.task_id}"


# =============================================================================
# 4. Ground Truth Normalization
# =============================================================================

class TestGroundTruthNormalization:
    """Verifies that all 4,000 ground truth targets follow normalized boxed or scalar representations."""

    def test_all_4000_ground_truths_enclosed_in_boxed(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify that every single task in the 4,000 suite has ground truth enclosed in \\boxed{...}."""
        for t in all_4000_tasks:
            gt_str = str(t.ground_truth).strip()
            assert "\\boxed{" in gt_str or gt_str.startswith("\\boxed"), (
                f"Ground truth not in boxed format for {t.task_id}: '{gt_str}'"
            )

    def test_latex_boxed_extractability(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify extract_latex_boxed successfully parses all 4,000 ground truth targets."""
        for t in all_4000_tasks:
            extracted = extract_latex_boxed(str(t.ground_truth))
            assert extracted is not None, f"Failed to extract boxed value from {t.task_id}: '{t.ground_truth}'"
            assert len(extracted.strip()) > 0, f"Extracted empty value from {t.task_id}"

    def test_ground_truth_brace_balance(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify that curly braces are strictly balanced in all 4,000 ground truths."""
        for t in all_4000_tasks:
            gt = str(t.ground_truth)
            assert gt.count("{") == gt.count("}"), f"Unbalanced braces in {t.task_id}: '{gt}'"

    def test_ground_truth_whitespace_cleanliness(self, all_4000_tasks: List[BenchmarkTask]):
        """Verify ground truths do not have carriage return artifacts or untrimmed outer whitespace."""
        for t in all_4000_tasks:
            gt = str(t.ground_truth)
            assert "\r" not in gt, f"Carriage return in ground truth of {t.task_id}"
            assert gt == gt.strip(), f"Untrimmed whitespace in ground truth of {t.task_id}"
            extracted = extract_latex_boxed(gt)
            assert extracted is not None and len(extracted.strip()) > 0


# =============================================================================
# 5. Evaluator Compatibility & Self-Consistency
# =============================================================================

class TestEvaluatorCompatibilityAndSelfConsistency:
    """Verifies that ground-truth answers evaluate cleanly (score 1.0) with evaluate_task_result()."""

    def test_evaluator_math_symbolic_self_consistency(
        self,
        math_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
    ):
        """Verify math_symbolic evaluation self-consistency on representative tasks."""
        math_samples = [t for t in math_tasks_1000 if t.eval_type == "math_symbolic"][:15]
        putnam_samples = [t for t in putnam_tasks_1000 if t.eval_type == "math_symbolic"][:15]
        combined = math_samples + putnam_samples

        for task in combined:
            res = evaluate_task_result(task, task.ground_truth)
            assert res.score == 1.0, f"Failed self-eval for {task.task_id}: score={res.score}, diag={res.diagnostic_message}"
            assert res.is_correct is True

    def test_evaluator_float_tol_self_consistency(
        self,
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
    ):
        """Verify float_tol evaluation self-consistency on representative GSM8K & SVAMP tasks."""
        gsm_samples = gsm8k_tasks_1000[:15]
        svamp_samples = svamp_tasks_1000[:15]
        combined = gsm_samples + svamp_samples

        for task in combined:
            res = evaluate_task_result(task, task.ground_truth)
            assert res.score == 1.0, f"Failed self-eval for {task.task_id}: score={res.score}, diag={res.diagnostic_message}"
            assert res.is_correct is True

    def test_evaluator_fraction_self_consistency(self, putnam_tasks_1000: List[BenchmarkTask]):
        """Verify fraction evaluation self-consistency on Putnam tasks."""
        fraction_tasks = [t for t in putnam_tasks_1000 if t.eval_type == "fraction"][:15]
        assert len(fraction_tasks) > 0, "No fraction tasks found in PutnamBench"

        for task in fraction_tasks:
            res = evaluate_task_result(task, task.ground_truth)
            assert res.score == 1.0, f"Failed fraction self-eval for {task.task_id}: score={res.score}"
            assert res.is_correct is True

    def test_evaluator_exact_eval_type(self):
        """Verify exact evaluation strategy handles boxed and scalar answers."""
        exact_task = BenchmarkTask(
            task_id="exact_sample_001",
            benchmark_name="synthetic",
            query="State the cardinality.",
            ground_truth="\\boxed{42}",
            eval_type="exact",
            metadata={"category": "discrete_math"},
        )
        res_exact = evaluate_task_result(exact_task, "\\boxed{42}")
        assert res_exact.score == 1.0
        assert res_exact.is_correct is True

        # Incorrect candidate
        res_wrong = evaluate_task_result(exact_task, "\\boxed{999}")
        assert res_wrong.score == 0.0
        assert res_wrong.is_correct is False

    def test_evaluator_rejects_incorrect_answers_across_all_eval_types(
        self,
        math_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
    ):
        """Verify evaluator returns 0.0 for clearly incorrect candidate outputs."""
        sample_tasks = [
            math_tasks_1000[0],
            gsm8k_tasks_1000[0],
            svamp_tasks_1000[0],
            putnam_tasks_1000[0],
        ]
        for task in sample_tasks:
            res = evaluate_task_result(task, "\\boxed{999999999}")
            assert res.score == 0.0, f"Expected 0.0 for bogus answer on {task.task_id}, got {res.score}"
            assert res.is_correct is False


# =============================================================================
# 6. Dry-Run Evaluation on Sample Tasks
# =============================================================================

class TestDryRunEvaluation:
    """Executes end-to-end dry-run evaluations on representative samples across all 4 suites."""

    def test_dry_run_math_representative_sample(self, math_tasks_1000: List[BenchmarkTask]):
        """Execute dry-run evaluation on 10 MATH tasks verifying non-zero accuracy and zero schema errors."""
        samples = math_tasks_1000[:10]
        for t in samples:
            res = evaluate_task_result(t, t.ground_truth)
            assert res.is_correct is True
            assert res.score == 1.0
            assert res.execution_time_ms >= 0.0

    def test_dry_run_putnam_representative_sample(self, putnam_tasks_1000: List[BenchmarkTask]):
        """Execute dry-run evaluation on 10 Putnam tasks verifying non-zero accuracy and zero schema errors."""
        samples = putnam_tasks_1000[:10]
        for t in samples:
            res = evaluate_task_result(t, t.ground_truth)
            assert res.is_correct is True
            assert res.score == 1.0
            assert res.execution_time_ms >= 0.0

    def test_dry_run_gsm8k_representative_sample(self, gsm8k_tasks_1000: List[BenchmarkTask]):
        """Execute dry-run evaluation on 10 GSM8K tasks verifying non-zero accuracy and zero schema errors."""
        samples = gsm8k_tasks_1000[:10]
        for t in samples:
            res = evaluate_task_result(t, t.ground_truth)
            assert res.is_correct is True
            assert res.score == 1.0
            assert res.execution_time_ms >= 0.0

    def test_dry_run_svamp_representative_sample(self, svamp_tasks_1000: List[BenchmarkTask]):
        """Execute dry-run evaluation on 10 SVAMP tasks verifying non-zero accuracy and zero schema errors."""
        samples = svamp_tasks_1000[:10]
        for t in samples:
            res = evaluate_task_result(t, t.ground_truth)
            assert res.is_correct is True
            assert res.score == 1.0
            assert res.execution_time_ms >= 0.0

    def test_dry_run_with_cot_wrapped_model_responses(
        self,
        math_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
    ):
        """Verify evaluate_task_result isolates boxed answers from verbose Chain-of-Thought prose."""
        for task in [math_tasks_1000[2], gsm8k_tasks_1000[2]]:
            cot_response = (
                f"We analyze the problem step-by-step.\n"
                f"Step 1: Compute the initial value.\n"
                f"Step 2: Apply the transformation.\n"
                f"Finally, the required result is {task.ground_truth}."
            )
            res = evaluate_task_result(task, cot_response)
            assert res.score == 1.0, f"Failed to isolate boxed answer from CoT for {task.task_id}"
            assert res.is_correct is True


# =============================================================================
# 7. Stratification Integrity
# =============================================================================

class TestStratificationIntegrity:
    """Verifies domain and difficulty stratification distributions across all four datasets."""

    def test_hendrycks_math_all_7_subjects_covered(self, math_tasks_1000: List[BenchmarkTask]):
        """Verify Hendrycks MATH covers all 7 core mathematical subjects."""
        expected_subjects = {
            "Algebra",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Number Theory",
            "Prealgebra",
            "Precalculus",
        }
        actual_subjects = {t.metadata.get("subject") or t.subdiscipline for t in math_tasks_1000}
        assert actual_subjects == expected_subjects, f"Subject mismatch: {actual_subjects ^ expected_subjects}"

        # Verify balanced distribution (each subject has at least 80 samples)
        counts = Counter(t.metadata.get("subject") or t.subdiscipline for t in math_tasks_1000)
        for subj, cnt in counts.items():
            assert cnt >= 80, f"Subject {subj} under-represented with {cnt} samples"

    def test_hendrycks_math_all_5_difficulty_levels_covered(self, math_tasks_1000: List[BenchmarkTask]):
        """Verify Hendrycks MATH covers difficulty Levels 1 through 5."""
        levels = {t.metadata.get("level") for t in math_tasks_1000}
        assert levels == {1, 2, 3, 4, 5}, f"Level coverage incomplete: {levels}"

        # Verify each level has at least 50 samples
        level_counts = Counter(t.metadata.get("level") for t in math_tasks_1000)
        for lvl, cnt in level_counts.items():
            assert cnt >= 50, f"Level {lvl} under-represented with {cnt} samples"

    def test_putnam_all_7_collegiate_subdisciplines_covered(self, putnam_tasks_1000: List[BenchmarkTask]):
        """Verify PutnamBench covers all 7 collegiate subdisciplines."""
        expected_categories = {
            "algebra",
            "analysis",
            "combinatorics",
            "geometry",
            "linear_algebra",
            "number_theory",
            "probability",
        }
        actual_categories = {t.metadata.get("category") or t.subdiscipline for t in putnam_tasks_1000}
        assert actual_categories == expected_categories, f"Category mismatch: {actual_categories ^ expected_categories}"

        # Verify each collegiate subdiscipline has at least 70 samples
        cat_counts = Counter(t.metadata.get("category") or t.subdiscipline for t in putnam_tasks_1000)
        for cat, cnt in cat_counts.items():
            assert cnt >= 70, f"Subdiscipline {cat} under-represented with {cnt} samples"

    def test_putnam_historical_year_span(self, putnam_tasks_1000: List[BenchmarkTask]):
        """Verify PutnamBench problems span 1962 through 2024."""
        years = [t.metadata.get("year") for t in putnam_tasks_1000 if t.metadata.get("year")]
        assert len(years) == 1000, "Some Putnam tasks lack year metadata"
        assert min(years) <= 1965, f"Earliest year too late: {min(years)}"
        assert max(years) >= 2020, f"Latest year too early: {max(years)}"
        assert 1962 in years and 2024 in years, "Years 1962 or 2024 missing from Putnam span"

    def test_gsm8k_all_integer_targets_and_multi_step_reasoning(self, gsm8k_tasks_1000: List[BenchmarkTask]):
        """Verify GSM8K tasks have exact integer targets and multi-step reasoning distribution."""
        for t in gsm8k_tasks_1000:
            int_target = t.metadata.get("integer_target")
            assert isinstance(int_target, int), f"Non-integer target for {t.task_id}: {int_target}"
            steps = t.metadata.get("reasoning_steps")
            assert isinstance(steps, int) and steps >= 2, f"Invalid reasoning steps for {t.task_id}: {steps}"

        step_counts = [t.metadata.get("reasoning_steps") for t in gsm8k_tasks_1000]
        avg_steps = sum(step_counts) / len(step_counts)
        assert 2.5 <= avg_steps <= 5.0, f"Average reasoning steps out of expected range: {avg_steps}"

    def test_svamp_all_4_arithmetic_operation_categories(self, svamp_tasks_1000: List[BenchmarkTask]):
        """Verify SVAMP covers 4 arithmetic operation categories (Addition, Subtraction, Multiplication, Common-Division)."""
        expected_operations = {"Addition", "Subtraction", "Multiplication", "Common-Division"}
        types = {t.metadata.get("type") or t.metadata.get("category") for t in svamp_tasks_1000}
        assert expected_operations.issubset(types), f"Missing SVAMP operation categories: {expected_operations - types}"

        type_counts = Counter(t.metadata.get("type") or t.metadata.get("category") for t in svamp_tasks_1000)
        for op in expected_operations:
            assert type_counts[op] >= 50, f"SVAMP category {op} under-represented with {type_counts[op]} samples"


# =============================================================================
# 8. Backward Compatibility & Test Invariants
# =============================================================================

class TestBackwardCompatibilityAndInvariants:
    """Verifies that legacy loader behavior and repository test invariants remain strictly preserved."""

    def test_math_loader_default_50_tasks(self):
        """Verify MATHLoader() with default arguments loads exactly 50 tasks from math_tasks.jsonl."""
        loader = MATHLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 50, f"Expected default 50 tasks for MATHLoader, got {len(tasks)}"

    def test_putnam_loader_default_50_tasks(self):
        """Verify PutnamBenchLoader() with default arguments loads exactly 50 tasks from putnam_tasks.jsonl."""
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks()
        assert len(tasks) == 50, f"Expected default 50 tasks for PutnamBenchLoader, got {len(tasks)}"

    def test_legacy_suite_invariant_50_50_350_is_450(self):
        """Verify the legacy invariant in test_adversarial_m1_m2.py (50 + 50 + 350 == 450) is preserved."""
        math_tasks = MATHLoader().load_tasks()
        putnam_tasks = PutnamBenchLoader().load_tasks()
        lila_tasks = LilaLoader().load_tasks()

        assert len(math_tasks) == 50
        assert len(putnam_tasks) == 50
        assert len(lila_tasks) == 350

        total_legacy = len(math_tasks) + len(putnam_tasks) + len(lila_tasks)
        assert total_legacy == 450, f"Expected 450 total legacy tasks, got {total_legacy}"

    def test_legacy_boundary_limit_clamping(self):
        """Verify that default MATHLoader and PutnamBenchLoader clamp limit=500 to 50."""
        math_tasks = MATHLoader().load_tasks(limit=500)
        assert len(math_tasks) == 50

        putnam_tasks = PutnamBenchLoader().load_tasks(limit=500)
        assert len(putnam_tasks) == 50

    def test_gsm8k_positional_split_compatibility(self):
        """Verify GSM8KLoader accepts positional split argument for backwards compatibility."""
        loader = GSM8KLoader("test")
        assert loader.split == TaskSplit.TEST
        tasks = loader.load_tasks(limit=5)
        assert len(tasks) == 5

    def test_all_loaders_support_get_task_and_manifest(
        self,
        math_tasks_1000: List[BenchmarkTask],
        putnam_tasks_1000: List[BenchmarkTask],
        gsm8k_tasks_1000: List[BenchmarkTask],
        svamp_tasks_1000: List[BenchmarkTask],
    ):
        """Verify get_task() and get_manifest() across all 4 expanded loaders."""
        math_loader = MATHLoader(use_1000=True)
        putnam_loader = PutnamBenchLoader(use_1000=True)
        gsm8k_loader = GSM8KLoader()
        svamp_loader = SVAMPLoader()

        # get_task
        assert math_loader.get_task(math_tasks_1000[0].task_id).task_id == math_tasks_1000[0].task_id
        assert putnam_loader.get_task(putnam_tasks_1000[0].task_id).task_id == putnam_tasks_1000[0].task_id
        assert gsm8k_loader.get_task(gsm8k_tasks_1000[0].task_id).task_id == gsm8k_tasks_1000[0].task_id
        assert svamp_loader.get_task(svamp_tasks_1000[0].task_id).task_id == svamp_tasks_1000[0].task_id

        # get_manifest
        for loader in (math_loader, putnam_loader, gsm8k_loader, svamp_loader):
            manifest = loader.get_manifest()
            assert isinstance(manifest, dict)
            assert manifest["total_tasks"] == 1000


# =============================================================================
# 9. Adversarial & Boundary Robustness
# =============================================================================

class TestAdversarialIntegrity:
    """Adversarial and boundary stress tests verifying error handling, escaping, and negative limits."""

    def test_get_task_nonexistent_id_raises_keyerror(self):
        """Verify get_task() raises KeyError when task_id does not exist."""
        math_loader = MATHLoader(use_1000=True)
        with pytest.raises(KeyError):
            math_loader.get_task("nonexistent_task_id_999999")

        svamp_loader = SVAMPLoader()
        with pytest.raises(KeyError):
            svamp_loader.get_task("nonexistent_svamp_id_999999")

    def test_negative_or_zero_limit_returns_empty_list(self):
        """Verify that limit<=0 returns an empty list across all loaders."""
        assert MATHLoader(use_1000=True).load_tasks(limit=0) == []
        assert MATHLoader(use_1000=True).load_tasks(limit=-1) == []
        assert PutnamBenchLoader(use_1000=True).load_tasks(limit=0) == []
        assert PutnamBenchLoader(use_1000=True).load_tasks(limit=-5) == []
        assert GSM8KLoader().load(limit=0) == []
        assert GSM8KLoader().load(limit=-10) == []
        assert SVAMPLoader().load(limit=0) == []
        assert SVAMPLoader().load(limit=-1) == []

    def test_evaluator_malformed_unclosed_boxed_does_not_crash(self, math_tasks_1000: List[BenchmarkTask]):
        """Verify that malformed LaTeX candidates (e.g. unclosed boxed) do not crash the evaluator."""
        task = math_tasks_1000[0]
        malformed_candidate = "\\boxed{unclosed_bracket"
        result = evaluate_task_result(task, malformed_candidate)
        assert isinstance(result, EvalResult)
        assert result.score == 0.0
        assert result.is_correct is False

    def test_evaluator_gibberish_and_special_characters_handling(self, svamp_tasks_1000: List[BenchmarkTask]):
        """Verify evaluator handles arbitrary control characters and special strings gracefully."""
        task = svamp_tasks_1000[0]
        weird_candidates = [
            "",
            "   ",
            "\\null\\undefined\\error",
            "!@#$%^&*()_+~`|}{[]:;?><,./",
            "None",
            "NaN",
            "Infinity",
        ]
        for candidate in weird_candidates:
            res = evaluate_task_result(task, candidate)
            assert isinstance(res, EvalResult)
            assert res.score == 0.0
            assert res.is_correct is False
