"""
tests.unit.test_adversarial_m1_m2
==================================
Empirical Adversarial Stress-Testing Suite for Milestones M1 and M2.

Stress-tests:
1. HardwareMonitor:
   - Zero / microsecond duration executions and sub-millisecond start/stop
   - Rapid sequential start/stop cycles (50 iterations)
   - Extreme memory allocation & deallocation churn and RSS tracking
   - Concurrent multi-threaded start, sample_current, and stop calls
   - GPU simulation: missing GPU, NVML failure exceptions, corrupted nvidia-smi CLI outputs
   - Energy trapezoidal integration under rapid power fluctuations and non-monotonic timestamps
   - Negative and near-zero sample_interval_s bounds
   - Idempotent stop() and sample_current() prior to start()
2. ValueExtractor:
   - Deeply nested LaTeX expressions & nested \\boxed{...} / \\fbox{...}
   - Malformed LaTeX: unclosed braces, unclosed boxed, extra closing braces
   - Markdown fences, JSON blocks, think tag isolation
   - Conflicting numbers in prose, distracting quantities, rightmost resolution
   - Empty strings, whitespace, Unicode zero-width characters, large input buffers (100k chars)
   - Expected type conditioning (set, float_tol, fraction, exact)
   - Empty boxed expressions (\\boxed{})
   - Complex quadratic formula extraction
   - Equation tail and anchor interaction with downstream SymPy evaluation
3. SympyMathEvaluator:
   - Pathological and extreme formulas, high powers (2^32, 2^64), continuous fractions
   - Division by zero and singularities (e.g. 1/0, 0/0, tan(pi/2), log(0))
   - Complex numbers with SymPy imaginary unit 'I' and Euler's formula
   - Set equivalence (multisets, order permutations, symbolic elements)
   - Rational vs decimal representations and tolerance thresholds
   - Trigonometric, hyperbolic, and factorial identities (cosh^2 - sinh^2 == 1, sin(x+y), cos(3x))
   - Polynomial factorization: (x^4 - y^4) == (x^2 + y^2)*(x + y)*(x - y)
4. Dataset Loaders (MATHLoader, PutnamBenchLoader, LilaLoader):
   - Boundary limits (limit=0, limit=-5, limit=1, limit=500 clamped)
   - Non-existent and empty category filters
   - Case-insensitivity and whitespace padding in subject/category names
   - Invalid split validation (ValueError) vs valid split transitions
   - Global task_id uniqueness and get_task lookup invariants
   - Missing fixture fallback and manifest structure
"""

import math
import os
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import sympy

from nemo_eval.datasets.base import BenchmarkTask, TaskSplit
from nemo_eval.datasets.lila import LilaLoader
from nemo_eval.datasets.math import MATHLoader
from nemo_eval.datasets.putnam import PutnamBenchLoader
from nemo_eval.eval.base import EvalResult
from nemo_eval.eval.math_eval import (
    SympyMathEvaluator,
    check_algebraic_equivalence,
    check_fraction_equivalence,
    check_set_and_interval_equivalence,
    evaluate_math_expression,
    extract_latex_boxed,
    normalize_latex_expression,
    parse_math_to_sympy,
)
from nemo_eval.telemetry.extractor import ValueExtractor
from nemo_eval.telemetry.monitor import HardwareMetrics, HardwareMonitor


# ===========================================================================
# 1. ADVERSARIAL HARDWARE MONITOR STRESS TESTS
# ===========================================================================

class TestAdversarialHardwareMonitor:
    """Empirical adversarial challenge suite for HardwareMonitor telemetry."""

    def test_zero_duration_instantaneous_stop(self):
        """Stress-test zero/sub-millisecond execution duration."""
        mon = HardwareMonitor(sample_interval_s=0.01)
        mon.start()
        # Immediately stop without sleeping
        metrics = mon.stop()
        assert metrics.duration_ms >= 0.0
        assert metrics.peak_ram_mb > 0.0
        assert metrics.energy_joules >= 0.0
        # Verify JSON serialization does not produce NaN/inf
        d = metrics.to_dict()
        assert isinstance(d["duration_ms"], float)
        assert isinstance(d["peak_ram_mb"], float)
        assert isinstance(d["energy_joules"], float)
        assert not math.isnan(d["duration_ms"])
        assert not math.isinf(d["duration_ms"])

    def test_sample_interval_clamping(self):
        """Verify negative or near-zero sample_interval_s is safely clamped to >= 0.001."""
        mon_neg = HardwareMonitor(sample_interval_s=-1.0)
        assert mon_neg.sample_interval_s >= 0.001

        mon_zero = HardwareMonitor(sample_interval_s=0.0)
        assert mon_zero.sample_interval_s >= 0.001

    def test_stop_before_start_idempotence(self):
        """Verify calling stop() or sample_current() before start() returns safe defaults."""
        mon = HardwareMonitor()
        m_sample = mon.sample_current()
        assert m_sample.duration_ms == 0.0
        assert m_sample.energy_joules == 0.0

        m_stop = mon.stop()
        assert m_stop.duration_ms == 0.0
        assert m_stop.energy_joules == 0.0

    def test_rapid_start_stop_cycles(self):
        """Stress-test 50 rapid sequential start/stop cycles."""
        mon = HardwareMonitor(sample_interval_s=0.005)
        for _ in range(50):
            mon.start()
            m = mon.stop()
            assert m.duration_ms >= 0.0
            assert m.peak_ram_mb > 0.0

    def test_concurrent_multithreaded_monitor_operations(self):
        """Stress-test thread-safety with 12 threads calling start, sample_current, and stop concurrently."""
        mon = HardwareMonitor(sample_interval_s=0.005)
        exceptions: List[Exception] = []

        def worker(worker_id: int):
            try:
                for _ in range(10):
                    if worker_id % 3 == 0:
                        mon.start()
                    elif worker_id % 3 == 1:
                        s = mon.sample_current()
                        assert s.peak_ram_mb >= 0.0
                    else:
                        m = mon.stop()
                        assert m.duration_ms >= 0.0
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        # Final cleanup
        final = mon.stop()
        assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"
        assert final.peak_ram_mb > 0.0

    def test_multiple_independent_monitor_instances(self):
        """Stress-test 10 concurrent independent HardwareMonitor instances in parallel threads."""
        results: List[HardwareMetrics] = []
        errors: List[Exception] = []

        def run_independent_monitor():
            try:
                with HardwareMonitor(sample_interval_s=0.01) as m:
                    time.sleep(0.02)
                    sample = m.sample_current()
                    assert sample.duration_ms > 0.0
                results.append(sample)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_independent_monitor) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)

        assert len(errors) == 0, f"Errors in concurrent monitors: {errors}"
        assert len(results) == 10

    def test_memory_allocation_churn_peak_tracking(self):
        """Verify that peak_ram_mb monotonically tracks peak RAM despite large allocation and immediate deletion."""
        mon = HardwareMonitor(sample_interval_s=0.005)
        mon.start()

        # Step 1: Record baseline RAM
        base_ram = mon.sample_current().peak_ram_mb

        # Step 2: Allocate 20MB buffer
        big_block = bytearray(20 * 1024 * 1024)
        time.sleep(0.03)
        sample_allocated = mon.sample_current()

        # Step 3: Deallocate buffer
        del big_block
        time.sleep(0.03)
        final = mon.stop()

        assert sample_allocated.peak_ram_mb >= base_ram
        assert final.peak_ram_mb >= sample_allocated.peak_ram_mb

    def test_gpu_simulation_nvml_exceptions_graceful_recovery(self):
        """Simulate pynvml raising various NVML errors (e.g. GPU lost, uninitialized)."""
        mon = HardwareMonitor(enable_gpu=True)
        mon._gpu_available = True
        mon._gpu_backend = "pynvml"
        mon._nvml_handle = MagicMock()

        # Mock pynvml module functions to raise exceptions
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = Exception("NVMLError: GPU is lost")
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("NVMLError: Not Supported")

        with patch("nemo_eval.telemetry.monitor.pynvml", mock_pynvml):
            vram, power = mon._sample_gpu()
            assert vram == 0.0
            assert power == 0.0

    def test_gpu_simulation_nvidia_smi_corrupted_output(self):
        """Simulate nvidia-smi returning garbage / malformed CSV or timing out."""
        mon = HardwareMonitor(enable_gpu=True)
        mon._gpu_available = True
        mon._gpu_backend = "nvidia_smi"
        mon._nvidia_smi_path = "fake_nvidia_smi"

        # Case 1: Malformed output
        with patch("subprocess.check_output", return_value=b"invalid,output,format,extra\n"):
            vram, power = mon._sample_gpu()
            assert isinstance(vram, float)
            assert isinstance(power, float)

        # Case 2: Non-numeric garbage
        with patch("subprocess.check_output", return_value=b"N/A, [Not Supported]\n"):
            vram, power = mon._sample_gpu()
            assert vram == 0.0
            assert power == 0.0

        # Case 3: Subprocess timeout exception
        with patch("subprocess.check_output", side_effect=Exception("TimeoutExpired")):
            vram, power = mon._sample_gpu()
            assert vram == 0.0
            assert power == 0.0

    def test_energy_trapezoidal_integration_extreme_power_spikes(self):
        """Stress-test numerical energy integration with 100 power samples and rapid fluctuations."""
        mon = HardwareMonitor(enable_gpu=False)
        mon._gpu_available = True

        # Generate 100 alternating power samples: 50W to 350W
        samples = []
        base_t = 1000.0
        expected_energy = 0.0

        for i in range(100):
            t = base_t + i * 0.1  # dt = 0.1s
            p = 50.0 if i % 2 == 0 else 350.0
            samples.append((t, p))
            if i > 0:
                dt = 0.1
                p_prev = samples[i - 1][1]
                expected_energy += 0.5 * (p_prev + p) * dt

        mon._power_samples = samples
        computed_energy = mon._compute_energy_joules(base_t + 99 * 0.1)
        assert pytest.approx(computed_energy, rel=1e-4) == expected_energy

    def test_energy_zero_samples_and_negative_time_protection(self):
        """Verify _compute_energy_joules returns 0.0 for empty samples or past timestamps."""
        mon = HardwareMonitor(enable_gpu=False)
        mon._gpu_available = True
        mon._power_samples = []
        assert mon._compute_energy_joules(10.0) == 0.0

        mon._power_samples = [(100.0, 50.0)]
        # Timestamp earlier than start
        assert mon._compute_energy_joules(50.0) == 0.0


# ===========================================================================
# 2. ADVERSARIAL VALUE EXTRACTOR STRESS TESTS
# ===========================================================================

class TestAdversarialValueExtractor:
    """Empirical adversarial challenge suite for ValueExtractor answer extraction."""

    def test_deeply_nested_boxed_latex_expressions(self):
        """Test extraction from deeply nested fractions, roots, and nested boxed wrappers."""
        raw = r"The final calculation yields \boxed{\frac{\sqrt{\frac{\sqrt{\frac{a}{b}+1}}{c}+2}}{d}}."
        extracted = ValueExtractor.extract_value(raw)
        assert extracted == r"\frac{\sqrt{\frac{\sqrt{\frac{a}{b}+1}}{c}+2}}{d}"

    def test_nested_boxed_inside_boxed(self):
        """Test double-boxed expressions: \boxed{\boxed{42}}."""
        raw = r"\boxed{\boxed{42}}"
        extracted = ValueExtractor.extract_value(raw)
        assert extracted == "42"

    def test_empty_boxed_expression(self):
        """Test extraction from empty boxed expression: \boxed{}."""
        raw = r"\boxed{}"
        assert ValueExtractor.extract_value(raw) == ""

    def test_quadratic_formula_boxed(self):
        """Test quadratic formula in boxed."""
        raw = r"\boxed{\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}}"
        extracted = ValueExtractor.extract_value(raw)
        assert r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}" in extracted

    def test_malformed_unclosed_and_extra_braces(self):
        """Test extraction resilience against unclosed braces and malformed LaTeX."""
        # Unclosed \boxed{
        assert ValueExtractor.extract_value(r"\boxed{999") == "999"
        # Extra trailing braces
        assert ValueExtractor.extract_value(r"\boxed{42}}") == "42"
        # Multiple unclosed attempts followed by a valid boxed
        raw = r"Attempt 1: \boxed{failed ... Attempt 2: \boxed{12345}"
        assert ValueExtractor.extract_value(raw) == "12345"

    def test_markdown_code_fences_and_nested_json(self):
        """Test extraction from markdown fences, JSON blocks, and code tags."""
        raw_json_md = """
Here is my final response:
```json
{
    "step_count": 5,
    "final_answer": "3.14159"
}
```
"""
        assert ValueExtractor.extract_value(raw_json_md) == "3.14159"

        raw_latex_md = "```latex\n\\boxed{\\frac{7}{12}}\n```"
        assert ValueExtractor.extract_value(raw_latex_md) == r"\frac{7}{12}"

    def test_think_tag_isolation_with_distractor_boxed(self):
        """Verify that distractors inside <think> tags or intermediate steps pick the final answer."""
        raw = r"""
<think>
First I thought the answer was \boxed{10}.
Then I tried \boxed{20}.
Finally I realized that 2 + 2 = 4.
</think>
After careful reasoning, the final answer is \boxed{4}.
"""
        # ValueExtractor picks the rightmost / final boxed expression
        assert ValueExtractor.extract_value(raw) == "4"

    @pytest.mark.parametrize("prose,expected", [
        ("We started with 5 apples, bought 10 more, sold 3, giving a total of 12 apples.", "12"),
        ("Problem 1: Solve for x. The 3 steps give 4 equations and final result is 42.", "42"),
        ("There are 20 boys and 30 girls. Therefore, the answer is 50", "50"),
        ("Option A is 10, Option B is 20, Option C is 30. The correct choice is 30.", "30"),
    ])
    def test_conflicting_numbers_in_prose(self, prose, expected):
        """Test resolving target answer when multiple numbers/distractors appear in prose."""
        assert ValueExtractor.extract_value(prose) == expected

    def test_equation_tail_and_anchor_downstream_eval(self):
        """Test equation assignment extraction and verification with SymPy evaluator."""
        # Case 1: Pure equation assignment near end without anchor delimiter
        prose_eq = "After extensive algebraic transformations:\nans = -2"
        extracted_eq = ValueExtractor.extract_value(prose_eq)
        assert extracted_eq == "-2"

        # Case 2: Anchor with equation
        prose_anchor = "x = 5 is extraneous; the only valid solution is x = -2."
        extracted_anchor = ValueExtractor.extract_value(prose_anchor)
        assert "x = -2" in extracted_anchor
        # Verify extracted string evaluates successfully against ground truth -2
        eval_res = evaluate_math_expression(extracted_anchor, "-2", eval_type="math_symbolic")
        assert eval_res.is_correct is True
        assert eval_res.score == 1.0

    def test_large_input_string_performance(self):
        """Test regex performance and lack of catastrophic backtracking on 100,000 char strings."""
        large_prefix = "Step evaluation test repeating text. " * 3000  # ~108,000 chars
        full_text = large_prefix + "\nFinal Answer: 987654321"

        t0 = time.perf_counter()
        extracted = ValueExtractor.extract_value(full_text)
        elapsed = time.perf_counter() - t0

        assert extracted == "987654321"
        assert elapsed < 0.5, f"Extraction on 100k chars took too long: {elapsed:.3f}s"

    def test_unicode_zero_width_and_whitespace_edge_cases(self):
        """Test handling of Unicode whitespace, zero-width spaces, and control characters."""
        # Zero-width spaces (\u200b, \ufeff)
        raw = "\u200b\ufeff  \\boxed{42}  \u200c"
        assert ValueExtractor.extract_value(raw) == "42"

        # Tab, newline, non-breaking space
        raw_ws = "\t\n \u00a0 \\boxed{100} \r\n"
        assert ValueExtractor.extract_value(raw_ws) == "100"

    @pytest.mark.parametrize("input_val,expected_type,expected_out", [
        ("1, 2, 3", "set", "{1, 2, 3}"),
        ("{1, 2, 3}", "set", "{1, 2, 3}"),
        ("42", "exact", "42"),
        ("12.50", "float_tol", "12.50"),
        ("3/4", "fraction", "3/4"),
    ])
    def test_expected_type_conditioning(self, input_val, expected_type, expected_out):
        """Test post-processing conditioning based on expected_type hints."""
        assert ValueExtractor.extract_value(input_val, expected_type=expected_type) == expected_out


# ===========================================================================
# 3. ADVERSARIAL SYMPY MATH EVALUATOR STRESS TESTS
# ===========================================================================

class TestAdversarialSympyMathEvaluator:
    """Empirical adversarial challenge suite for SympyMathEvaluator."""

    def test_pathological_nested_fractions(self):
        """Test equivalence of continuous nested fractions against exact rational."""
        # 1 / (1 + 1 / (1 + 1 / (1 + 1/2))) = 1 / (1 + 1 / (1 + 2/3)) = 1 / (1 + 3/5) = 5/8
        nested_latex = r"\frac{1}{1 + \frac{1}{1 + \frac{1}{1 + \frac{1}{2}}}}"
        gold = "5/8"
        res = evaluate_math_expression(nested_latex, gold, eval_type="math_symbolic")
        assert res.is_correct is True
        assert res.score == 1.0

    def test_massive_powers_and_integers(self):
        """Test equivalence of large integer powers (2^32, 2^64)."""
        cand = "2^{32}"
        gold = "4294967296"
        res = evaluate_math_expression(cand, gold, eval_type="math_symbolic")
        assert res.is_correct is True

        cand64 = "2^{64}"
        gold64 = "18446744073709551616"
        res64 = evaluate_math_expression(cand64, gold64, eval_type="math_symbolic")
        assert res64.is_correct is True

    @pytest.mark.parametrize("pathological_expr", [
        "1/0",
        "0/0",
        r"\frac{x}{0}",
        r"\tan(\pi/2)",
        r"\frac{1}{x - x}",
        "log(0)",
        "sqrt(-1) / 0",
    ])
    def test_division_by_zero_and_singularities_handled_gracefully(self, pathological_expr):
        """Verify that mathematical singularities and division by zero never crash the engine."""
        res = evaluate_math_expression(pathological_expr, "42", eval_type="math_symbolic")
        assert res.is_correct is False
        assert res.score == 0.0
        assert res.diagnostic_message is not None

    @pytest.mark.parametrize("invalid_token", [
        "(((+++***",
        r"\undefined_macro_xyz{123}",
        "---",
        "***???",
        "function() -> None",
        "{[}]",
    ])
    def test_non_algebraic_tokens_and_syntax_errors(self, invalid_token):
        """Verify unparseable syntax or non-algebraic tokens fail gracefully."""
        res = evaluate_math_expression(invalid_token, "x + 1", eval_type="math_symbolic")
        assert res.is_correct is False
        assert res.score == 0.0

    def test_complex_numbers_and_eulers_identity(self):
        """Test complex number algebra, imaginary units, and Euler's formula: exp(i*pi) + 1 == 0."""
        # (1 + I)^2 == 2*I
        assert check_algebraic_equivalence("(1 + I)^2", "2*I") is True

        # (1 + i)^2 == 1 + 2*i + i^2 (symbolic variable i)
        assert check_algebraic_equivalence("(1 + i)^2", "1 + 2*i + i^2") is True

        # exp(I * pi) + 1 == 0
        res = evaluate_math_expression(r"\exp(I \pi) + 1", "0", eval_type="math_symbolic")
        assert res.is_correct is True

    def test_multivariate_polynomial_identities(self):
        """Test complex 3-variable polynomial expansion: (a + b + c)^2."""
        cand = "(a + b + c)^2"
        gold = "a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a"
        assert check_algebraic_equivalence(cand, gold) is True
        res = evaluate_math_expression(cand, gold, eval_type="math_symbolic")
        assert res.is_correct is True

    def test_high_degree_polynomial_factorization(self):
        """Test (x - 1)^5 expansion equivalence."""
        cand = "(x - 1)^5"
        gold = "x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1"
        assert check_algebraic_equivalence(cand, gold) is True

    def test_multivariable_quartic_factorization(self):
        """Test x^4 - y^4 == (x^2 + y^2)*(x + y)*(x - y)."""
        cand = "x^4 - y^4"
        gold = "(x^2 + y^2)*(x + y)*(x - y)"
        assert check_algebraic_equivalence(cand, gold) is True

    @pytest.mark.parametrize("cand_set,gold_set,expected_match", [
        (r"\{3, 1, 4, 1, 5\}", r"\{1, 3, 4, 5\}", False),  # multiset count mismatch (two 1s vs one 1)
        (r"\{1, 1, 2, 3\}", r"\{1, 2, 1, 3\}", True),      # multiset count matches
        (r"\{\sqrt{9}, \frac{4}{2}, 2^3\}", r"\{3, 2, 8\}", True),  # symbolic element equivalence
        (r"\{x + 1, y - 1\}", r"\{y - 1, x + 1\}", True),  # variable elements
        (r"\{1, 2\}", r"\{1, 2, 3\}", False),               # cardinality mismatch
        (r"\{\}", r"\{\}", True),                           # empty sets
    ])
    def test_set_and_multiset_equivalence(self, cand_set, gold_set, expected_match):
        """Test mathematical set checking across permutations, multiset counts, and expressions."""
        is_equiv = check_set_and_interval_equivalence(cand_set, gold_set)
        assert is_equiv is expected_match
        res = evaluate_math_expression(cand_set, gold_set, eval_type="set")
        assert res.is_correct is expected_match

    def test_rational_vs_decimal_tolerance(self):
        """Test rational fraction vs floating point decimal conversions."""
        # 1/3 vs 0.333333 (within 1% tolerance)
        assert check_fraction_equivalence("1/3", "0.333333") is True
        assert check_fraction_equivalence("355/113", "3.1415929") is True
        # 1/3 vs 0.50 (outside tolerance)
        assert check_fraction_equivalence("1/3", "0.50") is False

    def test_advanced_trigonometric_and_hyperbolic_identities(self):
        """Test hyperbolic and multi-angle trigonometric identities."""
        # cosh(x)^2 - sinh(x)^2 == 1
        assert check_algebraic_equivalence(r"\cosh^2(x) - \sinh^2(x)", "1") is True
        # sin(x + y) == sin(x)*cos(y) + cos(x)*sin(y)
        assert check_algebraic_equivalence(r"\sin(x + y)", r"\sin(x)*\cos(y) + \cos(x)*\sin(y)") is True
        # cos(3*x) == 4*cos(x)^3 - 3*cos(x)
        assert check_algebraic_equivalence(r"\cos(3*x)", r"4*\cos^3(x) - 3*\cos(x)") is True


# ===========================================================================
# 4. ADVERSARIAL BENCHMARK DATASET LOADERS STRESS TESTS
# ===========================================================================

class TestAdversarialDatasetLoaders:
    """Empirical adversarial challenge suite for MATHLoader, PutnamBenchLoader, and LilaLoader."""

    @pytest.mark.parametrize("limit,expected_len", [
        (0, 0),
        (-1, 0),
        (-999, 0),
        (1, 1),
        (25, 25),
        (50, 50),
        (500, 50),  # clamped to max 50
    ])
    def test_math_loader_boundary_limits(self, limit, expected_len):
        """Test boundary limits for MATHLoader."""
        loader = MATHLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_len

    @pytest.mark.parametrize("limit,expected_len", [
        (0, 0),
        (-1, 0),
        (1, 1),
        (25, 25),
        (50, 50),
        (500, 50),  # clamped to max 50
    ])
    def test_putnam_loader_boundary_limits(self, limit, expected_len):
        """Test boundary limits for PutnamBenchLoader."""
        loader = PutnamBenchLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_len

    @pytest.mark.parametrize("limit,expected_len", [
        (0, 0),
        (-1, 0),
        (1, 1),
        (50, 50),
        (350, 350),
        (1000, 350),  # clamped to max 350
    ])
    def test_lila_loader_boundary_limits(self, limit, expected_len):
        """Test boundary limits for LilaLoader."""
        loader = LilaLoader()
        tasks = loader.load_tasks(limit=limit)
        assert len(tasks) == expected_len

    def test_math_loader_nonexistent_subject_returns_empty(self):
        """Test filtering by a non-existent subject returns empty list without raising exception."""
        loader = MATHLoader(subject="Astrophysics")
        tasks = loader.load_tasks()
        assert len(tasks) == 0

    def test_putnam_loader_nonexistent_category_returns_empty(self):
        """Test filtering by a non-existent category returns empty list without raising exception."""
        loader = PutnamBenchLoader(category="Quantum_Computing")
        tasks = loader.load_tasks()
        assert len(tasks) == 0

    def test_lila_loader_nonexistent_subcategories(self):
        """Test non-existent subcategories return empty list."""
        loader_nonexistent = LilaLoader(subcategories=["non_existent_category"])
        assert len(loader_nonexistent.load_tasks()) == 0

    def test_case_insensitivity_and_whitespace_in_loader_filters(self):
        """Test whitespace and casing tolerance in category/subject filters."""
        math_tasks = MATHLoader(subject="  aLgEbRa  ").load_tasks()
        assert len(math_tasks) > 0
        assert all(t.subdiscipline == "Algebra" for t in math_tasks)

        putnam_tasks = PutnamBenchLoader(category="  REAL_ANALYSIS  ").load_tasks()
        assert len(putnam_tasks) > 0
        assert all(t.subdiscipline.lower().replace(" ", "_") == "real_analysis" for t in putnam_tasks)

        lila_tasks = LilaLoader(subcategories=["  CALCULUS  "]).load_tasks()
        assert len(lila_tasks) == 50
        assert all(t.subdiscipline.lower() == "calculus" for t in lila_tasks)

    @pytest.mark.parametrize("invalid_split", [
        "invalid_split_xyz",
        "12345",
        "eval_super",
        "training_set",
    ])
    def test_invalid_splits_raise_value_error(self, invalid_split):
        """Verify invalid split names raise ValueError across all loaders."""
        with pytest.raises(ValueError, match="Unknown split"):
            MATHLoader().load(split=invalid_split)

        with pytest.raises(ValueError, match="Unknown split"):
            PutnamBenchLoader().load(split=invalid_split)

        with pytest.raises(ValueError, match="Unknown split"):
            LilaLoader().load(split=invalid_split)

    def test_global_task_id_uniqueness_across_all_benchmarks(self):
        """Verify that every single task ID across all 3 benchmark suites is globally unique."""
        math_tasks = MATHLoader().load_tasks()
        putnam_tasks = PutnamBenchLoader().load_tasks()
        lila_tasks = LilaLoader().load_tasks()

        all_ids = [t.task_id for t in math_tasks + putnam_tasks + lila_tasks]
        unique_ids = set(all_ids)

        assert len(all_ids) == 50 + 50 + 350 == 450
        assert len(unique_ids) == 450, f"Duplicate task IDs detected: {len(all_ids) - len(unique_ids)}"

    def test_get_task_exact_retrieval_and_invalid_id_keyerror(self):
        """Verify get_task works for all IDs and raises KeyError for invalid IDs."""
        math_loader = MATHLoader()
        first_math = math_loader.load_tasks(limit=1)[0]
        assert math_loader.get_task(first_math.task_id).task_id == first_math.task_id
        with pytest.raises(KeyError):
            math_loader.get_task("math_nonexistent_9999")

        putnam_loader = PutnamBenchLoader()
        first_putnam = putnam_loader.load_tasks(limit=1)[0]
        assert putnam_loader.get_task(first_putnam.task_id).task_id == first_putnam.task_id
        with pytest.raises(KeyError):
            putnam_loader.get_task("putnam_nonexistent_9999")

        lila_loader = LilaLoader()
        first_lila = lila_loader.load_tasks(limit=1)[0]
        assert lila_loader.get_task(first_lila.task_id).task_id == first_lila.task_id
        with pytest.raises(KeyError):
            lila_loader.get_task("lila_nonexistent_9999")

    def test_missing_fixture_file_fallback_generator(self, tmp_path):
        """Verify that loaders cleanly fall back to inline generation if JSONL fixture is missing."""
        empty_dir = str(tmp_path / "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)

        m_loader = MATHLoader(dataset_root=empty_dir)
        tasks_math = m_loader.load_tasks()
        assert len(tasks_math) == 50

        p_loader = PutnamBenchLoader(dataset_root=empty_dir)
        tasks_putnam = p_loader.load_tasks()
        assert len(tasks_putnam) == 50

        l_loader = LilaLoader(dataset_root=empty_dir)
        tasks_lila = l_loader.load_tasks()
        assert len(tasks_lila) == 350
