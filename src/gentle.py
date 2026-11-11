"""Keep training polite on a laptop: cool-down pauses on GPU heat, a rest between steps, low CPU priority.

Standard library only. Reads the GPU with `nvidia-smi` (ships with the NVIDIA driver).
"""
from __future__ import annotations

import subprocess
import sys
import time


def gpu_temp() -> int | None:
    """GPU core temperature in °C, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def cool_down(max_temp: int, resume_temp: int, log=print, read=gpu_temp, sleep=time.sleep,
              poll_s: int = 15, max_wait_s: int = 600) -> float:
    """If the GPU is at/above max_temp, wait until it drops to resume_temp (at most max_wait_s). Returns seconds waited."""
    t = read()
    if t is None or t < max_temp:
        return 0.0
    log(f"[gentle] GPU {t}°C >= {max_temp}°C, pausing until {resume_temp}°C")
    waited = 0.0
    # Cap the wait: something else (games, external monitor) may keep the GPU above resume_temp.
    while t is not None and t > resume_temp and waited < max_wait_s:
        sleep(poll_s)
        waited += poll_s
        t = read()
    log(f"[gentle] GPU {t}°C, resuming after {waited:.0f}s")
    return waited


def lower_priority() -> None:
    """Run below normal CPU priority so the desktop stays responsive."""
    if sys.platform == "win32":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        # Keep Windows from sleeping while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); display may still turn off.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    else:
        import os
        os.nice(10)


if __name__ == "__main__":  # self-check: python -m src.gentle
    temps = iter([80, 72, 66, 64])
    waited = cool_down(75, 65, log=lambda m: None, read=lambda: next(temps), sleep=lambda s: None, poll_s=15)
    assert waited == 45, waited
    assert cool_down(75, 65, read=lambda: 60) == 0.0
    assert cool_down(75, 65, read=lambda: None) == 0.0
    assert cool_down(75, 65, log=lambda m: None, read=lambda: 80, sleep=lambda s: None, max_wait_s=60) == 60
    print("gentle self-check ok, live GPU temp:", gpu_temp(), "°C")
