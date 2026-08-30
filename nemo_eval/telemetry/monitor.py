"""
nemo_eval.telemetry.monitor
----------------------------
Real-time hardware resource telemetry monitor for RAM, GPU VRAM, power, and energy.

Captures:
- Process tree RAM (RSS in MB) via psutil.
- GPU VRAM (MB) and power draw (Watts) via pynvml (NVML) or nvidia-smi CLI fallback.
- Energy consumption (Joules) via numerical integration (trapezoidal rule).
- Graceful degradation returning 0.0 values with gpu_available=False if no GPU is present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

try:
    import pynvml
    _PYNVML_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:
    pynvml = None
    _PYNVML_IMPORT_ERROR = _err


@dataclass
class HardwareMetrics:
    """Snapshot and aggregated hardware resource telemetry metrics."""
    duration_ms: float = 0.0
    peak_ram_mb: float = 0.0
    gpu_vram_mb: float = 0.0
    gpu_power_watts: float = 0.0
    energy_joules: float = 0.0
    gpu_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a rounded dictionary representation."""
        return {
            "duration_ms": round(self.duration_ms, 3),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "gpu_vram_mb": round(self.gpu_vram_mb, 2),
            "gpu_power_watts": round(self.gpu_power_watts, 2),
            "energy_joules": round(self.energy_joules, 4),
            "gpu_available": self.gpu_available,
        }


class HardwareMonitor:
    """
    Background hardware resource telemetry monitor.

    Uses a lightweight background daemon thread to sample RAM and GPU metrics
    at a configurable interval (default 0.05s / 50ms).
    """

    # Class-level cache for GPU driver discovery to eliminate redundant subprocess queries
    _gpu_discovered: bool = False
    _cached_gpu_backend: str = "none"
    _cached_nvml_handle: Any = None
    _cached_nvidia_smi_path: Optional[str] = None
    _cached_gpu_available: bool = False
    _discovery_lock = threading.Lock()

    def __init__(self, sample_interval_s: float = 0.05, enable_gpu: bool = True):
        self.sample_interval_s = max(0.001, sample_interval_s)
        self.enable_gpu = enable_gpu

        self._process = psutil.Process()
        self._gpu_available: bool = False
        self._gpu_backend: str = "none"  # "pynvml", "nvidia_smi", "none"
        self._nvml_handle: Any = None
        self._nvidia_smi_path: Optional[str] = None

        self._running: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._peak_ram_mb: float = 0.0
        self._peak_gpu_vram_mb: float = 0.0
        self._last_gpu_power_watts: float = 0.0
        self._power_samples: List[Tuple[float, float]] = []  # (monotonic_time, power_watts)

        if self.enable_gpu:
            self._init_gpu()

    # ------------------------------------------------------------------ #
    # GPU Discovery & Querying
    # ------------------------------------------------------------------ #

    @classmethod
    def _detect_gpu_cached(cls) -> Tuple[str, Any, Optional[str], bool]:
        """Discover GPU backend once and cache findings globally across monitor instances."""
        with cls._discovery_lock:
            if cls._gpu_discovered:
                return (
                    cls._cached_gpu_backend,
                    cls._cached_nvml_handle,
                    cls._cached_nvidia_smi_path,
                    cls._cached_gpu_available,
                )

            # 1. Try pynvml
            if pynvml is not None:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        # Probe to confirm working
                        pynvml.nvmlDeviceGetMemoryInfo(handle)
                        pynvml.nvmlDeviceGetPowerUsage(handle)
                        cls._cached_gpu_backend = "pynvml"
                        cls._cached_nvml_handle = handle
                        cls._cached_nvidia_smi_path = None
                        cls._cached_gpu_available = True
                        cls._gpu_discovered = True
                        return (
                            cls._cached_gpu_backend,
                            cls._cached_nvml_handle,
                            cls._cached_nvidia_smi_path,
                            cls._cached_gpu_available,
                        )
                except Exception:
                    pass

            # 2. Try nvidia-smi CLI
            smi_path = shutil.which("nvidia-smi")
            if smi_path:
                try:
                    out = subprocess.check_output(
                        [smi_path, "--query-gpu=memory.used,power.draw", "--format=csv,nounits,noheader"],
                        stderr=subprocess.DEVNULL,
                        timeout=0.5,
                    ).decode("utf-8", errors="ignore").strip()
                    if out:
                        parts = [p.strip() for p in out.splitlines()[0].split(",")]
                        if len(parts) >= 2:
                            cls._cached_gpu_backend = "nvidia_smi"
                            cls._cached_nvml_handle = None
                            cls._cached_nvidia_smi_path = smi_path
                            cls._cached_gpu_available = True
                            cls._gpu_discovered = True
                            return (
                                cls._cached_gpu_backend,
                                cls._cached_nvml_handle,
                                cls._cached_nvidia_smi_path,
                                cls._cached_gpu_available,
                            )
                except Exception:
                    pass

            # 3. Fallback: No GPU available
            cls._cached_gpu_backend = "none"
            cls._cached_nvml_handle = None
            cls._cached_nvidia_smi_path = None
            cls._cached_gpu_available = False
            cls._gpu_discovered = True
            return (
                cls._cached_gpu_backend,
                cls._cached_nvml_handle,
                cls._cached_nvidia_smi_path,
                cls._cached_gpu_available,
            )

    def _init_gpu(self) -> None:
        """Initialize instance GPU state from global cache."""
        backend, handle, smi_path, available = self._detect_gpu_cached()
        self._gpu_backend = backend
        self._nvml_handle = handle
        self._nvidia_smi_path = smi_path
        self._gpu_available = available

        if self._gpu_available:
            vram_mb, power_w = self._sample_gpu()
            self._peak_gpu_vram_mb = vram_mb
            self._last_gpu_power_watts = power_w
        else:
            self._peak_gpu_vram_mb = 0.0
            self._last_gpu_power_watts = 0.0

    def _sample_gpu(self) -> Tuple[float, float]:
        """Query current GPU VRAM (MB) and power draw (Watts)."""
        if not self._gpu_available:
            return 0.0, 0.0

        if self._gpu_backend == "pynvml" and self._nvml_handle is not None and pynvml is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                vram_mb = mem.used / (1024.0 * 1024.0)
                power_w = power_mw / 1000.0
                return vram_mb, power_w
            except Exception:
                pass

        if self._gpu_backend == "nvidia_smi" and self._nvidia_smi_path:
            try:
                out = subprocess.check_output(
                    [self._nvidia_smi_path, "--query-gpu=memory.used,power.draw", "--format=csv,nounits,noheader"],
                    stderr=subprocess.DEVNULL,
                    timeout=0.5,
                ).decode("utf-8", errors="ignore").strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 2:
                        return float(parts[0]), float(parts[1])
            except Exception:
                pass

        return 0.0, 0.0

    # ------------------------------------------------------------------ #
    # RAM Querying (Process Tree)
    # ------------------------------------------------------------------ #

    def _sample_ram(self) -> float:
        """Query total RAM (MB) for current process and all active child processes."""
        try:
            total_rss = self._process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return 0.0
        except Exception:
            return 0.0

        try:
            children = self._process.children(recursive=True)
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception:
                    continue
        except Exception:
            pass

        return total_rss / (1024.0 * 1024.0)

    # ------------------------------------------------------------------ #
    # Energy Integration
    # ------------------------------------------------------------------ #

    def _compute_energy_joules(self, current_time: float) -> float:
        """
        Compute energy in Joules using trapezoidal numerical integration over power samples.
        If only 1 sample or uniform power, fallback to avg_power * elapsed_seconds.
        """
        if not self._gpu_available or not self._power_samples:
            return 0.0

        if len(self._power_samples) == 1:
            t0, p0 = self._power_samples[0]
            dt = max(0.0, current_time - t0)
            return p0 * dt

        energy = 0.0
        for i in range(len(self._power_samples) - 1):
            t_a, p_a = self._power_samples[i]
            t_b, p_b = self._power_samples[i + 1]
            dt = max(0.0, t_b - t_a)
            energy += 0.5 * (p_a + p_b) * dt

        # Add tail interval from last sample to current_time
        t_last, p_last = self._power_samples[-1]
        dt_tail = max(0.0, current_time - t_last)
        energy += p_last * dt_tail

        return max(0.0, energy)

    # ------------------------------------------------------------------ #
    # Background Sampling Loop
    # ------------------------------------------------------------------ #

    def _sample_loop(self) -> None:
        """Background thread polling RAM and GPU metrics at sample_interval_s."""
        while not self._stop_event.is_set():
            t_now = time.monotonic()
            ram_mb = self._sample_ram()
            vram_mb, power_w = self._sample_gpu()

            with self._lock:
                if ram_mb > self._peak_ram_mb:
                    self._peak_ram_mb = ram_mb
                if vram_mb > self._peak_gpu_vram_mb:
                    self._peak_gpu_vram_mb = vram_mb
                if self._gpu_available:
                    self._last_gpu_power_watts = power_w
                    self._power_samples.append((t_now, power_w))

            self._stop_event.wait(self.sample_interval_s)

    # ------------------------------------------------------------------ #
    # Public Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start hardware telemetry background monitor."""
        with self._lock:
            if self._running:
                return

            self._stop_event.clear()
            self._start_time = time.monotonic()
            self._stop_time = 0.0

            # Initial sample
            initial_ram = self._sample_ram()
            self._peak_ram_mb = initial_ram

            vram_mb, power_w = self._sample_gpu()
            self._peak_gpu_vram_mb = vram_mb
            self._last_gpu_power_watts = power_w
            self._power_samples = [(self._start_time, power_w)] if self._gpu_available else []

            self._running = True

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def sample_current(self) -> HardwareMetrics:
        """Take a thread-safe snapshot of live hardware metrics without stopping the monitor."""
        now = time.monotonic()
        with self._lock:
            if self._start_time > 0.0:
                duration_ms = max(0.0, (now - self._start_time) * 1000.0)
            else:
                duration_ms = 0.0

            # Get latest RAM reading and update peak monotonically
            current_ram = self._sample_ram()
            if current_ram > self._peak_ram_mb:
                self._peak_ram_mb = current_ram

            # Energy
            energy_j = self._compute_energy_joules(now)

            return HardwareMetrics(
                duration_ms=round(duration_ms, 3),
                peak_ram_mb=round(self._peak_ram_mb, 2),
                gpu_vram_mb=round(self._peak_gpu_vram_mb, 2),
                gpu_power_watts=round(self._last_gpu_power_watts, 2),
                energy_joules=round(energy_j, 4),
                gpu_available=self._gpu_available,
            )

    def stop(self) -> HardwareMetrics:
        """Stop background sampling thread and return finalized HardwareMetrics."""
        with self._lock:
            if not self._running and self._stop_time > 0.0:
                # Already stopped: return cached final metrics
                duration_ms = max(0.0, (self._stop_time - self._start_time) * 1000.0) if self._start_time > 0 else 0.0
                energy_j = self._compute_energy_joules(self._stop_time)
                return HardwareMetrics(
                    duration_ms=round(duration_ms, 3),
                    peak_ram_mb=round(self._peak_ram_mb, 2),
                    gpu_vram_mb=round(self._peak_gpu_vram_mb, 2),
                    gpu_power_watts=round(self._last_gpu_power_watts, 2),
                    energy_joules=round(energy_j, 4),
                    gpu_available=self._gpu_available,
                )

            self._running = False
            self._stop_time = time.monotonic()
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)

        # Final sample
        final_ram = self._sample_ram()
        vram_mb, power_w = self._sample_gpu()

        with self._lock:
            if final_ram > self._peak_ram_mb:
                self._peak_ram_mb = final_ram
            if vram_mb > self._peak_gpu_vram_mb:
                self._peak_gpu_vram_mb = vram_mb
            if self._gpu_available:
                self._last_gpu_power_watts = power_w
                self._power_samples.append((self._stop_time, power_w))

            duration_ms = max(0.0, (self._stop_time - self._start_time) * 1000.0) if self._start_time > 0 else 0.0
            energy_j = self._compute_energy_joules(self._stop_time)

            return HardwareMetrics(
                duration_ms=round(duration_ms, 3),
                peak_ram_mb=round(self._peak_ram_mb, 2),
                gpu_vram_mb=round(self._peak_gpu_vram_mb, 2),
                gpu_power_watts=round(self._last_gpu_power_watts, 2),
                energy_joules=round(energy_j, 4),
                gpu_available=self._gpu_available,
            )

    # ------------------------------------------------------------------ #
    # Context Manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> HardwareMonitor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
