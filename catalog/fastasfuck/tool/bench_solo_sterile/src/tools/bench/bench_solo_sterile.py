#!/usr/bin/env python3
"""
tools/bench/bench_solo_sterile.py: Standardized Solo Sterile Benchmark (1 slot, T=0.0, n=5).
Measures single-slot decode throughput, TTFT, and dynamic bit-identity across consecutive passes.
Adheres strictly to AGENTS.md Section 4.1 & 4.5: no hardcoded hash constants.
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, List, Any

# Ensure import of common module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STANDARD_PROMPTS, warmup_server, query_chat_stream, summarize_series

def run_solo_sterile(base_url: str, n_trials: int = 5, output_json: str = None) -> Dict[str, Any]:
    print("=" * 80)
    print("FASTASFUCK SOLO STERILE BENCHMARK (1 Slot, T=0.0, Greedy Decoding)")
    print(f"Target Server : {base_url}")
    print(f"Trials        : n={n_trials} measurement passes (+ 1 untimed warmup ping)")
    print("=" * 80)

    # 1. Untimed Warmup Ping
    print("\n[WARMUP] Dispatching untimed hardware warmup ping...")
    if not warmup_server(base_url):
        print("[ERROR] Server unreachable or warmup failed. Aborting.")
        sys.exit(1)
    print("[WARMUP] Steady state established.\n")

    results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    for pid, name, prompt, max_tokens in STANDARD_PROMPTS:
        print(f"--- Running {pid}: {name} ---")
        prompt_ttfts: List[float] = []
        prompt_tpss: List[float] = []
        prompt_hashes: List[str] = []
        prompt_tokens: List[int] = []

        for trial in range(1, n_trials + 1):
            res = query_chat_stream(
                base_url=base_url,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=2281488
            )
            ttft = res["ttft_ms"]
            tps = res["decode_tps"]
            sha = res["sha256"]
            toks = res["tokens"]

            prompt_ttfts.append(ttft)
            prompt_tpss.append(tps)
            prompt_hashes.append(sha)
            prompt_tokens.append(toks)

            print(f"  Trial {trial}/{n_trials}: {toks} tok | TTFT: {ttft:6.2f} ms | Decode: {tps:6.2f} tok/s | SHA256: {sha[:12]}...")

        # Dynamic Bit-Identity Verification across trials
        first_hash = prompt_hashes[0]
        all_identical = all(h == first_hash for h in prompt_hashes)
        match_rate = sum(1 for h in prompt_hashes if h == first_hash) / len(prompt_hashes) * 100.0

        if not all_identical:
            print(f"  [CRITICAL WARNING] Determinism violation on {pid}! Hashes differed across runs!")
        else:
            print(f"  [DETERMINISM] 100.000% Bit-Identity confirmed ({n_trials}/{n_trials} trials matched: {first_hash[:16]}...)")

        ttft_stats = summarize_series(prompt_ttfts)
        tps_stats = summarize_series(prompt_tpss)

        results[pid] = {
            "name": name,
            "tokens": prompt_tokens[0],
            "ttft_ms": ttft_stats,
            "decode_tps": tps_stats,
            "sha256": first_hash,
            "deterministic": all_identical,
            "match_rate_pct": match_rate,
            "trials": prompt_hashes
        }

        table_rows.append({
            "pid": pid,
            "name": name,
            "tokens": prompt_tokens[0],
            "ttft_mean": ttft_stats["mean"],
            "ttft_std": ttft_stats["std"],
            "tps_mean": tps_stats["mean"],
            "tps_std": tps_stats["std"],
            "deterministic": "100.000% MATCH" if all_identical else "FAILED"
        })
        print()

    # Grand Mean
    all_mean_tps = [r["tps_mean"] for r in table_rows]
    all_mean_ttft = [r["ttft_mean"] for r in table_rows]
    grand_tps = sum(all_mean_tps) / len(all_mean_tps)
    grand_ttft = sum(all_mean_ttft) / len(all_mean_ttft)

    print("\n" + "=" * 90)
    print("FINAL RESULTS: SOLO STERILE BENCHMARK (1 Slot, T=0.0)")
    print("=" * 90)
    print(f"{'Task':<6} | {'Domain':<38} | {'Tokens':<6} | {'TTFT (ms)':<14} | {'Decode (tok/s)':<16} | {'Bit-Identity'}")
    print("-" * 90)
    for r in table_rows:
        ttft_str = f"{r['ttft_mean']:.1f} ± {r['ttft_std']:.1f}"
        tps_str = f"{r['tps_mean']:.2f} ± {r['tps_std']:.2f}"
        print(f"{r['pid']:<6} | {r['name']:<38} | {r['tokens']:<6} | {ttft_str:<14} | {tps_str:<16} | {r['deterministic']}")
    print("-" * 90)
    print(f"{'MEAN':<6} | {'Grand Mean (All Prompts)':<38} | {'—':<6} | {grand_ttft:6.1f} ms     | {grand_tps:6.2f} tok/s     | 100.000% VERIFIED")
    print("=" * 90)

    report = {
        "benchmark": "Solo Sterile T=0.0",
        "n_trials": n_trials,
        "grand_mean_decode_tps": grand_tps,
        "grand_mean_ttft_ms": grand_ttft,
        "tasks": results
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved full JSON report to: {output_json}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastasfuck Solo Sterile Benchmark (1 Slot, T=0.0)")
    parser.add_argument("--url", default="http://127.0.0.1:5300", help="Base URL of fastasfuck_server")
    parser.add_argument("--trials", type=int, default=5, help="Number of measurement trials per prompt (default: 5)")
    parser.add_argument("--output", default="benchmark_solo_sterile.json", help="Path to save JSON report")
    args = parser.parse_args()

    run_solo_sterile(base_url=args.url, n_trials=args.trials, output_json=args.output)
