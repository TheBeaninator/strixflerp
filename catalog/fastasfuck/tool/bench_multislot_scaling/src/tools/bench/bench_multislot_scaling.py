#!/usr/bin/env python3
"""
tools/bench/bench_multislot_scaling.py: Multi-Slot Concurrency Benchmark (1, 2, 4 slots).
Measures aggregate throughput, per-slot speed, and ITL under real concurrent multi-agent load.
Adheres strictly to AGENTS.md Section 4.1 & 4.2 (all active slots under real concurrent load).
"""

import sys
import os
import json
import time
import argparse
import concurrent.futures
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STANDARD_PROMPTS, warmup_server, query_chat_stream, summarize_series

def run_concurrent_pass(base_url: str, n_slots: int, prompt_indices: List[int], max_tokens: int = 256) -> Dict[str, Any]:
    """Execute one parallel pass across n_slots concurrently."""
    prompts_to_run = [STANDARD_PROMPTS[idx % len(STANDARD_PROMPTS)] for idx in prompt_indices[:n_slots]]

    t0 = time.time()
    slot_results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as executor:
        futures = [
            executor.submit(
                query_chat_stream,
                base_url,
                p[2],
                max_tokens,
                0.6,
                2281488 + i
            )
            for i, p in enumerate(prompts_to_run)
        ]
        for f in concurrent.futures.as_completed(futures):
            slot_results.append(f.result())

    wall_sec = time.time() - t0
    total_tokens = sum(r["tokens"] for r in slot_results)
    agg_tps = total_tokens / wall_sec if wall_sec > 0 else 0.0
    slot_tpss = [r["decode_tps"] for r in slot_results]

    all_itls: List[float] = []
    for r in slot_results:
        all_itls.extend(r["itls_ms"])

    return {
        "n_slots": n_slots,
        "wall_sec": wall_sec,
        "total_tokens": total_tokens,
        "agg_tps": agg_tps,
        "slot_tps_mean": sum(slot_tpss) / len(slot_tpss) if slot_tpss else 0.0,
        "itl_stats": summarize_series(all_itls)
    }

def run_multislot_benchmark(base_url: str, n_trials: int = 3, output_json: str = None) -> Dict[str, Any]:
    print("=" * 80)
    print("FASTASFUCK MULTI-SLOT CONCURRENCY BENCHMARK (1, 2, 4 Slots, T=0.6)")
    print(f"Target Server : {base_url}")
    print(f"Trials        : n={n_trials} measurement trials per configuration (+ warmup)")
    print("=" * 80)

    print("\n[WARMUP] Dispatching untimed hardware warmup ping...")
    if not warmup_server(base_url):
        print("[ERROR] Server unreachable or warmup failed. Aborting.")
        sys.exit(1)
    print("[WARMUP] Steady state established.\n")

    slot_configs = [1, 2, 4]
    results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    for n_slots in slot_configs:
        print(f"--- Testing Concurrency: {n_slots} Active Slot{'s' if n_slots > 1 else ''} ---")

        # Warmup iteration
        print(f"  Warmup iteration (discarded)...", end="", flush=True)
        _ = run_concurrent_pass(base_url, n_slots, list(range(n_slots)))
        print(" Done.")

        agg_tpss: List[float] = []
        slot_tpss: List[float] = []
        itl_p50s: List[float] = []
        itl_p95s: List[float] = []
        itl_p99s: List[float] = []

        for trial in range(1, n_trials + 1):
            prompt_indices = [(trial - 1 + s) % len(STANDARD_PROMPTS) for s in range(n_slots)]
            pass_res = run_concurrent_pass(base_url, n_slots, prompt_indices)

            agg_tps = pass_res["agg_tps"]
            slot_tps = pass_res["slot_tps_mean"]
            itls = pass_res["itl_stats"]

            agg_tpss.append(agg_tps)
            slot_tpss.append(slot_tps)
            itl_p50s.append(itls["p50"])
            itl_p95s.append(itls["p95"])
            itl_p99s.append(itls["p99"])

            print(f"  Trial {trial}/{n_trials}: Agg: {agg_tps:6.2f} tok/s | Slot Mean: {slot_tps:5.2f} tok/s | ITL p50: {itls['p50']:5.1f} ms | ITL p95: {itls['p95']:5.1f} ms")

        agg_stats = summarize_series(agg_tpss)
        slot_stats = summarize_series(slot_tpss)
        p50_mean = sum(itl_p50s) / len(itl_p50s)
        p95_mean = sum(itl_p95s) / len(itl_p95s)
        p99_mean = sum(itl_p99s) / len(itl_p99s)

        results[str(n_slots)] = {
            "n_slots": n_slots,
            "agg_tps": agg_stats,
            "slot_tps": slot_stats,
            "itl_p50_ms": p50_mean,
            "itl_p95_ms": p95_mean,
            "itl_p99_ms": p99_mean,
            "n_trials": n_trials
        }

        table_rows.append({
            "n_slots": n_slots,
            "agg_mean": agg_stats["mean"],
            "agg_std": agg_stats["std"],
            "slot_mean": slot_stats["mean"],
            "slot_std": slot_stats["std"],
            "itl_p50": p50_mean,
            "itl_p95": p95_mean,
            "itl_p99": p99_mean,
            "n_trials": n_trials
        })
        print()

    print("\n" + "=" * 95)
    print("FINAL RESULTS: MULTI-SLOT CONCURRENCY SCALING (T=0.6)")
    print("=" * 95)
    print(f"{'Slots':<8} | {'Aggregate Throughput (tok/s)':<30} | {'Per-Slot Throughput (tok/s)':<28} | {'ITL p50':<9} | {'ITL p95'}")
    print("-" * 95)
    for r in table_rows:
        agg_str = f"{r['agg_mean']:.2f} ± {r['agg_std']:.2f} (n={r['n_trials']})"
        slot_str = f"{r['slot_mean']:.2f} ± {r['slot_std']:.2f}"
        print(f"{r['n_slots']:<8} | {agg_str:<30} | {slot_str:<28} | {r['itl_p50']:5.1f} ms  | {r['itl_p95']:5.1f} ms")
    print("=" * 95)

    report = {
        "benchmark": "Multi-Slot Concurrency Scaling T=0.6",
        "results": results
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved full JSON report to: {output_json}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastasfuck Multi-Slot Concurrency Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:5300", help="Base URL of fastasfuck_server")
    parser.add_argument("--trials", type=int, default=3, help="Number of measurement trials (default: 3)")
    parser.add_argument("--output", default="benchmark_multislot_scaling.json", help="Path to save JSON report")
    args = parser.parse_args()

    run_multislot_benchmark(base_url=args.url, n_trials=args.trials, output_json=args.output)
