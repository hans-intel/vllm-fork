#!/usr/bin/env python3
"""Confirm the dist-sample fast path ACTUALLY ENGAGED during a run.

This is the check that would have caught the 0%-improvement bug: the feature
methods existed and env=1 was set, but the call sites were never wired, so the
stock sampler ran every step and perf was unchanged.

The fast path logs a one-time marker on its first invocation:
    [DIST_SAMPLE] fused gumbel+int64-maxreduce fast path ENGAGED

Usage:
    python3 check_engaged.py <run_log_dir_or_file> [<baseline_log_dir_or_file>]

Exit codes:
    0 = engaged (marker found)
    1 = NOT engaged (marker absent) -- feature is dead, perf result is meaningless
"""

import sys
from pathlib import Path

MARKER = "[DIST_SAMPLE]"
ENGAGED = "ENGAGED"


def find_logs(path: Path):
    if path.is_file():
        return [path]
    # Directory: check common log file names.
    candidates = []
    for name in ("run.log", "mlperf_log_detail.txt", "vllm.log"):
        p = path / name
        if p.exists():
            candidates.append(p)
    # Also any *.log in the dir.
    candidates += [p for p in path.glob("*.log") if p not in candidates]
    return candidates


def scan(path: Path):
    """Return (engaged: bool, dist_sample_env: str|None, lines: list[str])."""
    logs = find_logs(path)
    if not logs:
        print(f"  ✗ No log files found under {path}")
        return False, None, []

    engaged = False
    env_val = None
    marker_lines = []
    for log in logs:
        try:
            text = log.read_text(errors="replace")
        except Exception as e:
            print(f"  ⚠ Could not read {log}: {e}")
            continue
        for line in text.splitlines():
            if "VLLM_XPU_DIST_SAMPLE=" in line and "FUSED" not in line and "ALLGATHER" not in line:
                # Capture the value of the bare flag.
                for tok in line.split():
                    if tok.startswith("VLLM_XPU_DIST_SAMPLE="):
                        env_val = tok.split("=", 1)[1]
            if MARKER in line and ENGAGED in line:
                engaged = True
                marker_lines.append(line.strip())
    return engaged, env_val, marker_lines


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    print("=" * 70)
    print("Dist-Sample Runtime Engagement Check")
    print("=" * 70)

    target = Path(sys.argv[1])
    print(f"\nRun under test: {target}")
    engaged, env_val, lines = scan(target)
    print(f"  VLLM_XPU_DIST_SAMPLE = {env_val}")
    if engaged:
        print(f"  ✓ ENGAGED -- found marker:")
        for l in lines[:3]:
            print(f"      {l}")
    else:
        print(f"  ✗ NOT ENGAGED -- '{MARKER} ... {ENGAGED}' marker absent")
        if env_val == "1":
            print("    Env was set to 1 but the path never fired. Likely causes:")
            print("      - call sites not wired into execute_model/sample_tokens")
            print("      - batch ineligible (temp!=1, top_p/top_k set, spec-decode, logprobs)")
            print("      - TP=1 (path engages but no benefit -- check separately)")

    # Optional baseline: must NOT have engaged.
    if len(sys.argv) >= 3:
        base = Path(sys.argv[2])
        print(f"\nBaseline: {base}")
        b_engaged, b_env, _ = scan(base)
        print(f"  VLLM_XPU_DIST_SAMPLE = {b_env}")
        if b_engaged:
            print("  ⚠ Baseline ALSO engaged dist-sample -- not a clean baseline!")
        else:
            print("  ✓ Baseline did not engage dist-sample (clean)")

    print("")
    if engaged:
        print("✓ Fast path engaged. A flat perf result here is a REAL result")
        print("  (kernel/collective genuinely not faster), not a wiring bug.")
        return 0
    else:
        print("✗ Fast path NOT engaged. Any perf comparison is meaningless --")
        print("  the stock sampler ran. Fix wiring/eligibility before measuring.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
