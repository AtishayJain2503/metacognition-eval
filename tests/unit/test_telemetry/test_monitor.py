"""
Unit tests for nemo_eval.telemetry.monitor (Milestone M1 - Hardware Telemetry Subsystem).
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from nemo_eval.telemetry.monitor import HardwareMetrics, HardwareMonitor


class TestHardwareMetrics:
    """Test HardwareMetrics dataclass schema and serialization."""

    def test_default_values(self):
        m = HardwareMetrics()
        assert m.duration_ms == 0.0
        assert m.peak_ram_mb == 0.0
        assert m.gpu_vram_mb == 0.0
        assert m.gpu_power_watts == 0.0
        assert m.energy_joules == 0.0
        assert m.gpu_available is False

    def test_to_dict_rounding(self):
        m = HardwareMetrics(
            duration_ms=123.45678,
            peak_ram_mb=45.6789,
            gpu_vram_mb=1024.1234,
            gpu_power_watts=45.6789,
            energy_joules=12.345678,
            gpu_available=True,
        )
        d = m.to_dict()
        assert d["duration_ms"] == 123.457
        assert d["peak_ram_mb"] == 45.68
        assert d["gpu_vram_mb"] == 1024.12
        assert d["gpu_power_watts"] == 45.68
        assert d["energy_joules"] == 12.3457
        assert d["gpu_available"] is True


class TestHardwareMonitor:
    """Test HardwareMonitor background sampling, RAM tracking, and GPU fallback."""

    def test_monitor_lifecycle(self):
        mon = HardwareMonitor(sample_interval_s=0.01)
        mon.start()
        time.sleep(0.04)
        sample = mon.sample_current()
        assert sample.duration_ms > 0.0
        assert sample.peak_ram_mb > 0.0

        final = mon.stop()
        assert final.duration_ms >= sample.duration_ms
        assert final.peak_ram_mb >= sample.peak_ram_mb

    def test_context_manager(self):
        with HardwareMonitor(sample_interval_s=0.01) as mon:
            time.sleep(0.03)
            sample = mon.sample_current()
            assert sample.duration_ms > 0.0
            assert sample.peak_ram_mb > 0.0

    def test_ram_tracking_under_allocation(self):
        mon = HardwareMonitor(sample_interval_s=0.01)
        mon.start()
        # Allocate array
        arr = [i for i in range(300_000)]
        time.sleep(0.03)
        metrics = mon.stop()
        del arr
        assert metrics.peak_ram_mb > 10.0

    def test_energy_trapezoidal_integration(self):
        mon = HardwareMonitor(enable_gpu=False)
        mon._gpu_available = True
        mon._power_samples = [
            (10.0, 50.0),  # t=10s, P=50W
            (12.0, 70.0),  # t=12s, P=70W -> interval 2s, avg 60W -> 120 Joules
            (14.0, 90.0),  # t=14s, P=90W -> interval 2s, avg 80W -> 160 Joules
        ]
        # Current time 15.0s -> last interval 1s, P=90W -> 90 Joules
        # Total = 120 + 160 + 90 = 370 Joules
        energy = mon._compute_energy_joules(15.0)
        assert abs(energy - 370.0) < 1e-3

    def test_energy_single_sample_fallback(self):
        mon = HardwareMonitor(enable_gpu=False)
        mon._gpu_available = True
        mon._power_samples = [(10.0, 100.0)]
        energy = mon._compute_energy_joules(12.5)  # 2.5s * 100W = 250J
        assert abs(energy - 250.0) < 1e-3

    def test_gpu_disabled_or_unavailable_fallback(self):
        mon = HardwareMonitor(enable_gpu=False)
        mon.start()
        time.sleep(0.02)
        metrics = mon.stop()
        assert metrics.gpu_available is False
        assert metrics.gpu_vram_mb == 0.0
        assert metrics.gpu_power_watts == 0.0
        assert metrics.energy_joules == 0.0

    def test_idempotent_stop(self):
        mon = HardwareMonitor(sample_interval_s=0.01)
        mon.start()
        time.sleep(0.02)
        m1 = mon.stop()
        m2 = mon.stop()
        assert m1.duration_ms == m2.duration_ms
        assert m1.peak_ram_mb == m2.peak_ram_mb

    def test_sample_current_before_start(self):
        mon = HardwareMonitor()
        m = mon.sample_current()
        assert m.duration_ms == 0.0

    def test_sample_ram_handles_process_exceptions(self):
        mon = HardwareMonitor()
        with patch.object(mon._process, "memory_info", side_effect=Exception("Simulated error")):
            ram = mon._sample_ram()
            assert ram == 0.0
