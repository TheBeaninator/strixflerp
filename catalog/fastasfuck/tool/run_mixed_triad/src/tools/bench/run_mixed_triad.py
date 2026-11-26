#!/usr/bin/env python3
"""
tools/bench/run_mixed_triad.py: Automated runner for the Mixed Concurrency Triad:
chunk = 4096, chunk-shared in {1024, 512, 256}.
For each candidate:
1. Cleanly restarts fastasfuck_server via tmux.
2. Waits for /health to become online.
3. Executes bench_mixed_concurrency.py (with untimed ping + 16k warmup per profile).
4. Saves isolated JSON report.
5. Summarizes all three runs in a side-by-side comparative table.
"""

import sys
import os
import time
import json
import subprocess
import urllib.request
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SERVER_BIN = os.path.join(BASE_DIR, "build/strix/fastasfuck_server")
CONFIG_FILE = "/home/cheblan/fastasfuck-server.conf"
BASE_URL = "http://127.0.0.1:5300"
DOCS_DIR = os.path.join(BASE_DIR, "docs", "benchmarks", "1.0.0")

TRIAD = [1024, 512, 256]

def wait_for_server(timeout_sec: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False

def restart_server(chunk: int, chunk_shared: int):
    print(f"\n[ORCHESTRATOR] Stopping existing fastasfuck_server instance...")
    subprocess.run(["tmux", "send-keys", "-t", "strix", "C-c"], check=False)
    time.sleep(2.0)
    
    # Ensure killed
    res = subprocess.run(["pgrep", "-f", "fastasfuck_server"], capture_output=True, text=True)
    if res.stdout.strip():
        subprocess.run(["pkill", "-9", "-f", "fastasfuck_server"], check=False)
        time.sleep(2.0)

    print(f"[ORCHESTRATOR] Launching fastasfuck_server with --chunk {chunk} --chunk-shared {chunk_shared} in tmux 'strix'...")
    cmd = f"{SERVER_BIN} --config {CONFIG_FILE} --chunk {chunk} --chunk-shared {chunk_shared}"
    subprocess.run(["tmux", "send-keys", "-t", "strix", cmd, "C-m"], check=True)

    print("[ORCHESTRATOR] Waiting for server /health...")
    if not wait_for_server(60.0):
        raise RuntimeError("fastasfuck_server failed to become healthy within 60 seconds!")
    print("[ORCHESTRATOR] fastasfuck_server is online and healthy.\n")

def run_suite(output_json: str):
    script_path = os.path.join(BASE_DIR, "tools/bench/bench_mixed_concurrency.py")
    cmd = [
        sys.executable,
        script_path,
        "--url", BASE_URL,
        "--trials", "3",
        "--output", output_json
    ]
    print(f"[ORCHESTRATOR] Running: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"[ORCHESTRATOR] Completed in {elapsed:.1f}s. Report saved to: {output_json}\n")

def print_triad_summary():
    reports: Dict[int, Dict[str, Any]] = {}
    for cs in TRIAD:
        path = os.path.join(DOCS_DIR, f"benchmark_mixed_4096_{cs}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reports[cs] = json.load(f)

    if not reports:
        print("[SUMMARY] No reports found to summarize.")
        return

    print("\n" + "=" * 135)
    print("FASTASFUCK MIXED CONCURRENCY TRIAD COMPARISON (chunk = 4096, chunk-shared in {1024, 512, 256})")
    print("=" * 135)
    print(f"{'chunk-shared':<12} | {'Profile':<8} | {'Decode / Slot (tok/s)':<22} | {'Prefill Agg (tok/s)':<24} | {'Step p50':<9} | {'Step p95':<9} | {'Step p99':<9} | {'Max ITL':<9}")
    print("-" * 135)

    for cs in TRIAD:
        if cs not in reports:
            continue
        profs = reports[cs].get("profiles", {})
        for prof_key in ["3D_1P", "2D_2P", "1D_3P"]:
            if prof_key not in profs:
                continue
            p = profs[prof_key]
            dec_mean = p["decode_slot_tps"]["mean"]
            dec_std = p["decode_slot_tps"]["std"]
            pref_mean = p["prefill_agg_tps"]["mean"]
            pref_std = p["prefill_agg_tps"]["std"]
            p50 = p.get("step_p50_ms", 0.0)
            p95 = p.get("step_p95_ms", 0.0)
            p99 = p.get("step_p99_ms", 0.0)
            max_itl = p.get("step_max_ms", 0.0)
            
            p_short = prof_key.replace("_", "+")
            dec_str = f"{dec_mean:.2f} ± {dec_std:.2f}"
            pref_str = f"{pref_mean:.1f} ± {pref_std:.1f}"
            print(f"{cs:<12} | {p_short:<8} | {dec_str:<22} | {pref_str:<24} | {p50:5.1f} ms | {p95:5.1f} ms | {p99:5.1f} ms | {max_itl:5.1f} ms")
        print("-" * 135)
    print("=" * 135 + "\n")

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    for cs in TRIAD:
        output_file = os.path.join(DOCS_DIR, f"benchmark_mixed_4096_{cs}.json")
        print(f"\n{'#'*80}")
        print(f"# STAGE: Testing chunk = 4096, chunk-shared = {cs}")
        print(f"{'#'*80}")
        
        restart_server(chunk=4096, chunk_shared=cs)
        run_suite(output_file)

    print_triad_summary()

if __name__ == "__main__":
    main()
