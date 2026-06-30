#!/usr/bin/env python3
"""Measure decode-step latency from MLPerf logs."""

import json
import sys
from pathlib import Path
from statistics import median, mean, stdev


def extract_latency(log_dir):
    """Extract decode-step latencies from mlperf_log_detail.json."""
    detail_log = Path(log_dir) / "mlperf_log_detail.json"

    if not detail_log.exists():
        print(f"ERROR: {detail_log} not found")
        return None

    try:
        with open(detail_log) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {detail_log}: {e}")
        return None

    # Extract timestamps from detail entries
    timestamps = []
    for entry in data:
        if isinstance(entry, dict):
            if "detail" in entry and "unixts" in entry["detail"]:
                timestamps.append(entry["detail"]["unixts"])
            elif "unixts" in entry:
                timestamps.append(entry["unixts"])

    if not timestamps:
        print("ERROR: No timestamps found in log")
        return None

    # Sort timestamps and compute deltas (time between decode steps)
    timestamps.sort()
    deltas_ms = [(timestamps[i+1] - timestamps[i]) * 1000
                 for i in range(len(timestamps)-1)]

    if len(deltas_ms) < 10:
        print(f"WARNING: Only {len(deltas_ms)} delta measurements available (need >10)")
        return None

    # Skip first 20 steps (warmup) and last 20 (cooldown)
    warmup_skip = min(20, len(deltas_ms) // 4)
    cooldown_skip = min(20, len(deltas_ms) // 4)

    if len(deltas_ms) > warmup_skip + cooldown_skip:
        deltas_stable = deltas_ms[warmup_skip:-cooldown_skip]
    else:
        deltas_stable = deltas_ms

    if not deltas_stable:
        print("ERROR: Not enough stable measurements after warmup/cooldown skip")
        return None

    stats = {
        "count": len(deltas_stable),
        "median": median(deltas_stable),
        "mean": mean(deltas_stable),
        "min": min(deltas_stable),
        "max": max(deltas_stable),
        "stdev": stdev(deltas_stable) if len(deltas_stable) > 1 else 0,
    }

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: measure_latency.py <log_dir1> [log_dir2] [log_dir3]")
        print("  Compare latency across multiple runs")
        print("")
        print("Example:")
        print("  python3 measure_latency.py logs-baseline logs-fused logs-proto")
        sys.exit(1)

    log_dirs = sys.argv[1:]
    results = []

    print("=" * 70)
    print("Decode-Step Latency Measurement")
    print("=" * 70)
    print("")

    for log_dir in log_dirs:
        print(f"Analyzing {log_dir}...")
        stats = extract_latency(log_dir)

        if stats is None:
            print(f"  ✗ Failed to extract latency")
            results.append(None)
        else:
            print(f"  ✓ Measurements: {stats['count']}")
            print(f"    Median:  {stats['median']:7.2f} ms")
            print(f"    Mean:    {stats['mean']:7.2f} ms")
            print(f"    StdDev:  {stats['stdev']:7.2f} ms")
            print(f"    Range:   {stats['min']:7.2f} - {stats['max']:7.2f} ms")
            results.append(stats)
        print("")

    # Compute comparisons if we have 2+ results
    if len(results) >= 2 and all(r is not None for r in results):
        print("=" * 70)
        print("Latency Comparisons")
        print("=" * 70)
        baseline = results[0]
        for i, (log_dir, stats) in enumerate(zip(log_dirs[1:], results[1:]), 1):
            delta_ms = stats["median"] - baseline["median"]
            delta_pct = (delta_ms / baseline["median"]) * 100

            if delta_ms < 0:
                print(f"{log_dir}: {delta_ms:+7.2f} ms ({delta_pct:+6.2f}%) ✓ Faster")
            else:
                print(f"{log_dir}: {delta_ms:+7.2f} ms ({delta_pct:+6.2f}%) ⚠ Slower")
        print("")


if __name__ == "__main__":
    main()
