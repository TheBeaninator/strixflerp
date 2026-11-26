#!/usr/bin/env python3
"""
tools/bench/bench_chunk_pareto.py: 2-Phase Chunk & Chunk-Shared Pareto Sweep.
Phase A: Solo compute saturation knee for `chunk ∈ {1024, 2048, 4096, 8192, 16384}` on 16k and 64k prefill.
Phase B: Multi-agent closed-loop replay for candidate `chunk` × `chunk-shared ∈ {256, 512, 1024, 2048}`.
Adheres strictly to AGENTS.md Section 4.6.
"""

import sys
import os
import json
import time
import uuid
import subprocess
import argparse
import urllib.request
import concurrent.futures
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STANDARD_PROMPTS, warmup_server, query_chat_stream, query_chat_nonstream, summarize_series

CONF_PATH = "/home/cheblan/fastasfuck-server.conf"
SERVER_BIN = "/home/cheblan/fastasfuck-main/build/strix/fastasfuck_server"
URL = "http://127.0.0.1:5300"

CORPUS_CHUNK = (
    "In unified memory architectures, compute wavefronts share memory controllers with system bus agents. "
    "When prefill kernels saturate memory channels, low-latency speculative verification wavefronts experience "
    "inter-token latency jitter unless memory arbiters enforce deficit round-robin quantum chunking. "
)

def build_context(target_tokens: int, nonce: str = "") -> str:
    reps = max(1, target_tokens // 45)
    return f"Contention Workload [{nonce}]:\n\n" + (CORPUS_CHUNK * reps) + "\n\nSummarize scheduling."

def restart_server_tmux(chunk: int, chunk_shared: int):
    """Safely restarts fastasfuck_server inside tmux session 'strix' with explicit chunk settings."""
    print(f"\n[SERVER] Restarting server in tmux with --chunk {chunk} --chunk-shared {chunk_shared}...")
    
    # Send kill to old instance
    subprocess.run(["tmux", "send-keys", "-t", "strix", "C-c", ""], capture_output=True)
    time.sleep(1)
    subprocess.run(["pkill", "-9", "-f", "fastasfuck_server"], capture_output=True)
    time.sleep(2)
    
    # Start fresh session with bash
    subprocess.run(["tmux", "kill-session", "-t", "strix"], capture_output=True)
    time.sleep(1)
    subprocess.run(["tmux", "new-session", "-d", "-s", "strix", "bash"], check=True)
    
    # Launch server command
    cmd = f"{SERVER_BIN} --config {CONF_PATH} --chunk {chunk} --chunk-shared {chunk_shared}\n"
    subprocess.run(["tmux", "send-keys", "-t", "strix", cmd], check=True)
    
    # Poll /health until online
    for attempt in range(60):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"{URL}/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    print(f"[SERVER] Server is ONLINE and healthy ({attempt + 1}s).")
                    time.sleep(1)
                    warmup_server(URL, timeout=30.0)
                    print("[SERVER] Steady state established.\n")
                    return
        except Exception:
            pass
    raise RuntimeError("Server failed to start within 60 seconds!")

def run_phase_a_saturation(chunks: List[int] = None) -> Dict[str, Any]:
    """Phase A: Measure 1-slot prefill throughput on 16k and 64k tokens across chunk sizes."""
    if chunks is None:
        chunks = [1024, 2048, 4096, 8192, 16384]

    print("=" * 85)
    print("PHASE A: SOLO COMPUTE SATURATION KNEE (1 Slot, 16k & 64k Prefill)")
    print(f"Sweeping chunks: {chunks}")
    print("=" * 85)

    saturation_results: Dict[str, Any] = {}

    for c in chunks:
        restart_server_tmux(chunk=c, chunk_shared=c)

        # 16k test (n=3)
        p16k = build_context(16384, f"warm_{c}")
        # Warm PLE
        _ = query_chat_nonstream(URL, p16k, max_tokens=5, temperature=0.0)
        tps_16k: List[float] = []
        for trial in range(3):
            prompt = build_context(16384, f"pA_16k_{c}_{trial}_{uuid.uuid4().hex[:6]}")
            res = query_chat_nonstream(URL, prompt, max_tokens=5, temperature=0.0)
            tps_16k.append(res.get("prefill_tps", 0.0))

        # 64k test (n=3)
        p64k = build_context(65536, f"warm_{c}")
        _ = query_chat_nonstream(URL, p64k, max_tokens=5, temperature=0.0)
        tps_64k: List[float] = []
        for trial in range(3):
            prompt = build_context(65536, f"pA_64k_{c}_{trial}_{uuid.uuid4().hex[:6]}")
            res = query_chat_nonstream(URL, prompt, max_tokens=5, temperature=0.0)
            tps_64k.append(res.get("prefill_tps", 0.0))

        s16 = summarize_series(tps_16k)
        s64 = summarize_series(tps_64k)

        saturation_results[str(c)] = {
            "chunk": c,
            "16k_tps": s16,
            "64k_tps": s64
        }
        print(f"  chunk={c:<5} | 16k Prefill: {s16['mean']:6.1f} ± {s16['std']:4.1f} tok/s | 64k Prefill: {s64['mean']:6.1f} ± {s64['std']:4.1f} tok/s")

    print("\n" + "=" * 85)
    print("PHASE A SATURATION SUMMARY")
    print("=" * 85)
    print(f"{'Chunk Size':<12} | {'16k Prefill (tok/s)':<25} | {'64k Prefill (tok/s)'}")
    print("-" * 85)
    for c in chunks:
        r = saturation_results[str(c)]
        c_str = str(c)
        s16_str = f"{r['16k_tps']['mean']:.1f} ± {r['16k_tps']['std']:.1f}"
        s64_str = f"{r['64k_tps']['mean']:.1f} ± {r['64k_tps']['std']:.1f}"
        print(f"{c_str:<12} | {s16_str:<25} | {s64_str}")
    print("=" * 85 + "\n")
    return saturation_results

def run_phase_b_pareto(pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Phase B: 4-slot closed loop agentic replay for candidate (chunk, chunk-shared) pairs.
    Workload: 3 slots decoding (256 tokens) + 1 slot prefilling (16k context) + 1 short-turn tool call (50 tokens).
    Captures: Makespan (s), Turn E2E p95 (s), Short-Turn TTFT p95 (ms), Step ITL p99 (ms).
    """
    print("=" * 105)
    print("PHASE B: MULTI-AGENT CLOSED-LOOP REPLAY PARETO FRONTIER")
    print(f"Sweeping (chunk, chunk-shared) pairs: {pairs}")
    print("=" * 105)

    pareto_results: Dict[str, Any] = {}

    for c, c_shared in pairs:
        restart_server_tmux(chunk=c, chunk_shared=c_shared)

        makespans: List[float] = []
        turn_e2es: List[float] = []
        short_ttfts: List[float] = []
        step_p99s: List[float] = []

        n_trials = 3
        for trial in range(1, n_trials + 1):
            t_start = time.time()
            
            # Setup workload: 3 decoders + 1 prefill + 1 short tool call
            dec_prompts = [STANDARD_PROMPTS[i][2] for i in range(3)]
            pref_prompt = build_context(16384, f"pB_{c}_{c_shared}_{trial}_{uuid.uuid4().hex[:6]}")
            short_prompt = "Quick status ping: verify database schema integrity. Respond in 10 words."

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Launch prefill
                f_pref = executor.submit(query_chat_nonstream, URL, pref_prompt, 10, 0.0)
                time.sleep(0.1)
                
                # Launch 3 streaming decoders
                f_decs = [
                    executor.submit(query_chat_stream, URL, dec_prompts[i], 256, 0.6, 2281488 + i)
                    for i in range(3)
                ]
                
                # Launch short turn tool call after prefill has started
                time.sleep(0.3)
                f_short = executor.submit(query_chat_stream, URL, short_prompt, 50, 0.0, 9999)

                pref_res = f_pref.result()
                short_res = f_short.result()
                dec_res = [f.result() for f in f_decs]

            t_end = time.time()
            makespan = t_end - t_start
            makespans.append(makespan)
            short_ttfts.append(short_res["ttft_ms"])

            # Collect turn completion times across decoders
            for dr in dec_res:
                turn_e2es.append(dr["wall_sec"])

            # Collect step latencies
            all_steps: List[float] = []
            for dr in dec_res:
                all_steps.extend(dr.get("itls_ms", dr.get("step_intervals_ms", [])))
            s_stats = summarize_series(all_steps)
            step_p99s.append(s_stats["p99"])

            print(f"  ({c}, {c_shared}) Trial {trial}/{n_trials}: Makespan: {makespan:5.2f}s | Short TTFT: {short_res['ttft_ms']:6.1f}ms | Step p99: {s_stats['p99']:5.1f}ms")

        p_key = f"{c}_{c_shared}"
        pareto_results[p_key] = {
            "chunk": c,
            "chunk_shared": c_shared,
            "makespan_sec": summarize_series(makespans),
            "turn_e2e_sec": summarize_series(turn_e2es),
            "short_ttft_ms": summarize_series(short_ttfts),
            "step_p99_ms": summarize_series(step_p99s)
        }

    print("\n" + "=" * 105)
    print("PHASE B PARETO FRONTIER SUMMARY")
    print("=" * 105)
    print(f"{'(chunk, chunk-shared)':<24} | {'Total Makespan (s)':<22} | {'Turn E2E p95 (s)':<18} | {'Short TTFT p95 (ms)':<22} | {'Step ITL p99'}")
    print("-" * 105)
    for c, c_shared in pairs:
        r = pareto_results[f"{c}_{c_shared}"]
        pair_str = f"({c}, {c_shared})"
        make_str = f"{r['makespan_sec']['mean']:.2f} ± {r['makespan_sec']['std']:.2f}"
        e2e_str = f"{r['turn_e2e_sec']['p95']:.2f} s"
        ttft_str = f"{r['short_ttft_ms']['p95']:.1f} ms"
        itl_str = f"{r['step_p99_ms']['mean']:.1f} ms"
        print(f"{pair_str:<24} | {make_str:<22} | {e2e_str:<18} | {ttft_str:<22} | {itl_str}")
    print("=" * 105 + "\n")
    return pareto_results

def run_chunk_pareto_suite(output_json: str = None):
    # Phase A: Solo compute saturation knee sweep
    sat_results = run_phase_a_saturation(chunks=[1024, 2048, 4096, 8192, 16384])
    
    # Phase B: Sweeping pairs for the Pareto frontier
    pairs = [
        (4096, 256),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        (2048, 256),
        (2048, 512),
        (2048, 1024),
        (1024, 256),
    ]
    pareto_results = run_phase_b_pareto(pairs)

    # Restore default production config (chunk=4096, chunk_shared=1024)
    print("[SERVER] Restoring production configuration (chunk=4096, chunk-shared=1024)...")
    restart_server_tmux(chunk=4096, chunk_shared=1024)

    full_report = {
        "phase_a_saturation": sat_results,
        "phase_b_pareto": pareto_results
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved full Pareto report to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastasfuck 2-Phase Chunk & Chunk-Shared Pareto Sweep")
    parser.add_argument("--output", default="benchmark_chunk_pareto.json", help="Path to save report")
    args = parser.parse_args()

    run_chunk_pareto_suite(output_json=args.output)
