#!/usr/bin/env python3
"""
tools/bench/bench_mixed_concurrency.py: Mixed Concurrency Isolation Benchmark (3D+1P, 2D+2P, 1D+3P).
Measures decode throughput, decode step latency (p50/p95/p99/max), and aggregate prefill throughput strictly
during the mutual contention window.
Decoders loop continuously to eliminate EOS idle gaps. Token counts in window are derived from
server usage.completion_tokens proportionally by total character delta (content + reasoning_content).
Step latencies are computed strictly within each individual stream (never bridging across queries).
Adheres strictly to GPU time conservation laws and AGENTS.md Section 4.1, 4.2 & 4.6.
"""

import sys
import os
import json
import time
import uuid
import argparse
import threading
import urllib.request
import concurrent.futures
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STANDARD_PROMPTS, warmup_server, query_chat_nonstream, summarize_series

CORPUS_CHUNK = (
    "In unified memory architectures, compute wavefronts share memory controllers with system bus agents. "
    "When prefill kernels saturate memory channels, low-latency speculative verification wavefronts experience "
    "inter-token latency jitter unless memory arbiters enforce deficit round-robin quantum chunking. "
)

def build_uncached_prefill_prompt(target_tokens: int) -> str:
    reps = max(1, target_tokens // 45)
    nonce = uuid.uuid4().hex
    return f"Contention Workload [{nonce}]:\n\n" + (CORPUS_CHUNK * reps) + "\n\nSummarize scheduling."

def streaming_decoder_worker(
    base_url: str,
    prompt: str,
    temperature: float,
    seed: int,
    query_history: List[Dict[str, Any]],
    ready_event: threading.Event,
    stop_event: threading.Event
):
    """
    Streams tokens in a continuous loop until stop_event is set.
    Eliminates idle gaps if the model emits EOS (<|im_end|>) early.
    Records arrival timestamps and char deltas (content + reasoning_content),
    preserving exact usage.completion_tokens.
    """
    query_idx = 0
    while not stop_event.is_set():
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": "thinkingcap-qwen3.8-flash-next",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "seed": seed + (query_idx * 17)
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        t0 = time.time()
        chunk_events = [] # [(timestamp, char_count)]
        completion_tokens = 0
        total_chars = 0
        
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                        continue
                    t_now = time.time()
                    try:
                        frame = json.loads(line_str[6:])
                        if "choices" in frame and frame["choices"]:
                            delta = frame["choices"][0].get("delta", {})
                            content_txt = delta.get("content") or ""
                            reasoning_txt = delta.get("reasoning_content") or ""
                            n_chars = len(content_txt) + len(reasoning_txt)
                            if n_chars > 0:
                                total_chars += n_chars
                                chunk_events.append((t_now, n_chars))
                        if "usage" in frame and frame["usage"]:
                            completion_tokens = frame["usage"].get("completion_tokens", 0)
                    except Exception:
                        pass
                    
                    if not ready_event.is_set():
                        ready_event.set()
        except Exception:
            pass
            
        t1 = time.time()
        ready_event.set()
        
        if completion_tokens == 0:
            completion_tokens = len(chunk_events)
            
        query_history.append({
            "t_start": t0,
            "t_end": t1,
            "completion_tokens": completion_tokens,
            "total_chars": total_chars,
            "chunk_events": chunk_events
        })
        query_idx += 1

def run_mixed_profile(base_url: str, n_decode: int, n_prefill: int, prefill_tokens: int = 16384) -> Dict[str, Any]:
    """
    Executes a mixed profile (3D+1P, 2D+2P, 1D+3P) strictly within the mutual overlap window.
    Decoders loop to prevent early EOS termination.
    Token counts are derived from server usage frames proportionally by (content + reasoning_content) chars.
    Step latencies are computed strictly within each stream.
    """
    stop_event = threading.Event()
    decoder_ready_events = [threading.Event() for _ in range(n_decode)]
    decoder_histories: List[List[Dict[str, Any]]] = [[] for _ in range(n_decode)]
    decoder_threads = []

    # 1. Start decoders in loop mode
    for i in range(n_decode):
        prompt = STANDARD_PROMPTS[i % len(STANDARD_PROMPTS)][2]
        t = threading.Thread(
            target=streaming_decoder_worker,
            args=(base_url, prompt, 0.6, 2281488 + i, decoder_histories[i], decoder_ready_events[i], stop_event),
            daemon=True
        )
        decoder_threads.append(t)
        t.start()

    # Wait until all decoders have begun active generation (first token emitted)
    for r_ev in decoder_ready_events:
        r_ev.wait(timeout=30.0)

    # 2. Launch prefill workers simultaneously
    prefill_futures = []
    prefill_results = []
    
    t_pref_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_prefill) as executor:
        for _ in range(n_prefill):
            p_prompt = build_uncached_prefill_prompt(prefill_tokens)
            f = executor.submit(query_chat_nonstream, base_url, p_prompt, 5, 0.0, 120.0)
            prefill_futures.append(f)
        
        for f in concurrent.futures.as_completed(prefill_futures):
            prefill_results.append(f.result())
    t_pref_end = time.time()

    # 3. Stop decoders immediately after prefill completes
    stop_event.set()
    for t in decoder_threads:
        t.join(timeout=30.0)
    time.sleep(1.0) # Settle slot state before computing metrics and launching next trial

    window_sec = t_pref_end - t_pref_start
    total_prefill_tokens = sum(r.get("prompt_tokens", 0) for r in prefill_results)
    agg_prefill_tps = total_prefill_tokens / window_sec if window_sec > 0 else 0.0
    slot_prefill_tps = agg_prefill_tps / n_prefill if n_prefill > 0 else 0.0

    # 4. Count decode tokens emitted strictly within [t_pref_start, t_pref_end]
    decode_slot_tpss: List[float] = []
    all_step_intervals: List[float] = []

    for i in range(n_decode):
        slot_tokens_in_window = 0.0

        for q in decoder_histories[i]:
            if q["t_end"] < t_pref_start or q["t_start"] > t_pref_end:
                continue
            
            # Entire query within window
            if q["t_start"] >= t_pref_start and q["t_end"] <= t_pref_end:
                slot_tokens_in_window += q["completion_tokens"]
                q_chunks = [c_t for c_t, _ in q["chunk_events"]]
                for k in range(1, len(q_chunks)):
                    delta = (q_chunks[k] - q_chunks[k - 1]) * 1000.0
                    if delta >= 1.0:
                        all_step_intervals.append(delta)
            else:
                total_c = q["total_chars"]
                win_c = 0
                q_chunks_in_win = []
                for c_t, c_len in q["chunk_events"]:
                    if t_pref_start <= c_t <= t_pref_end:
                        win_c += c_len
                        q_chunks_in_win.append(c_t)

                if total_c > 0:
                    slot_tokens_in_window += q["completion_tokens"] * (win_c / total_c)
                else:
                    dur = max(0.001, q["t_end"] - q["t_start"])
                    overlap_dur = min(q["t_end"], t_pref_end) - max(q["t_start"], t_pref_start)
                    slot_tokens_in_window += q["completion_tokens"] * max(0.0, overlap_dur / dur)

                for k in range(1, len(q_chunks_in_win)):
                    delta = (q_chunks_in_win[k] - q_chunks_in_win[k - 1]) * 1000.0
                    if delta >= 1.0:
                        all_step_intervals.append(delta)

        tps = slot_tokens_in_window / window_sec if window_sec > 0 else 0.0
        decode_slot_tpss.append(tps)

    mean_decode_tps = sum(decode_slot_tpss) / len(decode_slot_tpss) if decode_slot_tpss else 0.0
    agg_decode_tps = sum(decode_slot_tpss)

    step_stats = summarize_series(all_step_intervals) if all_step_intervals else {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    return {
        "n_decode": n_decode,
        "n_prefill": n_prefill,
        "window_sec": window_sec,
        "total_prefill_tokens": total_prefill_tokens,
        "agg_prefill_tps": agg_prefill_tps,
        "slot_prefill_tps": slot_prefill_tps,
        "mean_decode_slot_tps": mean_decode_tps,
        "agg_decode_tps": agg_decode_tps,
        "step_stats": step_stats
    }

def run_mixed_suite(base_url: str, n_trials: int = 3, output_json: str = None) -> Dict[str, Any]:
    print("=" * 125)
    print("FASTASFUCK MIXED CONCURRENCY ISOLATION SUITE (AIRTIGHT MUTUAL CONTENTION)")
    print(f"Target Server : {base_url}")
    print(f"Trials        : n={n_trials} trials strictly inside the mutual contention window (decoders looped)")
    print("=" * 125)

    print("\n[WARMUP] Dispatching untimed hardware warmup ping...")
    if not warmup_server(base_url):
        print("[ERROR] Server unreachable or warmup failed. Aborting.")
        sys.exit(1)
    print("[WARMUP] Steady state established.\n")

    profiles = [
        (3, 1, "3D + 1P (3 Decoding, 1 Prefilling 16k)"),
        (2, 2, "2D + 2P (2 Decoding, 2 Prefilling 16k)"),
        (1, 3, "1D + 3P (1 Decoding, 3 Prefilling 16k)"),
    ]

    suite_results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    for n_dec, n_pref, label in profiles:
        print(f"--- Running Profile: {label} ---")
        
        # Warmup iteration
        print("  Warmup iteration (discarded)...", end="", flush=True)
        _ = run_mixed_profile(base_url, n_dec, n_pref, prefill_tokens=16384)
        print(" Done.")

        dec_tpss: List[float] = []
        pref_agg_tpss: List[float] = []
        step_p50s: List[float] = []
        step_p95s: List[float] = []
        step_p99s: List[float] = []
        step_maxs: List[float] = []
        windows: List[float] = []

        for trial in range(1, n_trials + 1):
            res = run_mixed_profile(base_url, n_dec, n_pref, prefill_tokens=16384)
            d_tps = res["mean_decode_slot_tps"]
            p_tps = res["agg_prefill_tps"]
            w_sec = res["window_sec"]
            p_toks = res["total_prefill_tokens"]
            steps = res["step_stats"]

            dec_tpss.append(d_tps)
            pref_agg_tpss.append(p_tps)
            step_p50s.append(steps["p50"])
            step_p95s.append(steps["p95"])
            step_p99s.append(steps["p99"])
            step_maxs.append(steps["max"])
            windows.append(w_sec)

            print(f"  Trial {trial}/{n_trials}: Decode Slot: {d_tps:5.2f} tok/s | Prefill Agg: {p_tps:6.1f} tok/s ({p_toks} tok in {w_sec:4.1f}s) | Step p95: {steps['p95']:5.1f} ms | Step p99: {steps['p99']:5.1f} ms | Max: {steps['max']:5.1f} ms")

        dec_stats = summarize_series(dec_tpss)
        pref_stats = summarize_series(pref_agg_tpss)
        p50_mean = sum(step_p50s) / len(step_p50s)
        p95_mean = sum(step_p95s) / len(step_p95s)
        p99_mean = sum(step_p99s) / len(step_p99s)
        max_step = max(step_maxs)
        win_mean = sum(windows) / len(windows)

        key = f"{n_dec}D_{n_pref}P"
        suite_results[key] = {
            "label": label,
            "n_decode": n_dec,
            "n_prefill": n_pref,
            "decode_slot_tps": dec_stats,
            "prefill_agg_tps": pref_stats,
            "step_p50_ms": p50_mean,
            "step_p95_ms": p95_mean,
            "step_p99_ms": p99_mean,
            "max_step_ms": max_step,
            "window_sec": win_mean
        }

        table_rows.append({
            "profile": f"{n_dec}D + {n_pref}P",
            "decode_slot_mean": dec_stats["mean"],
            "decode_slot_std": dec_stats["std"],
            "prefill_agg_mean": pref_stats["mean"],
            "prefill_agg_std": pref_stats["std"],
            "step_p50": p50_mean,
            "step_p95": p95_mean,
            "step_p99": p99_mean,
            "max_step": max_step,
            "window": win_mean
        })
        print()

    print("\n" + "=" * 125)
    print("FINAL RESULTS: MIXED CONCURRENCY ISOLATION (MUTUAL CONTENTION WINDOW)")
    print("=" * 125)
    print(f"{'Profile':<10} | {'Decode / Slot (tok/s)':<24} | {'Prefill Agg (tok/s)':<26} | {'Step p50':<10} | {'Step p95':<10} | {'Step p99':<10} | {'Max ITL'}")
    print("-" * 125)
    for r in table_rows:
        dec_str = f"{r['decode_slot_mean']:.2f} ± {r['decode_slot_std']:.2f}"
        pref_str = f"{r['prefill_agg_mean']:.1f} ± {r['prefill_agg_std']:.1f}"
        print(f"{r['profile']:<10} | {dec_str:<24} | {pref_str:<26} | {r['step_p50']:5.1f} ms   | {r['step_p95']:5.1f} ms   | {r['step_p99']:5.1f} ms   | {r['max_step']:5.1f} ms")
    print("=" * 125)

    report = {
        "benchmark": "Mixed Concurrency Suite (Mutual Contention Window)",
        "profiles": suite_results
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[REPORT] Saved full JSON report to: {output_json}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastasfuck Mixed Concurrency Isolation Suite")
    parser.add_argument("--url", default="http://127.0.0.1:5300", help="Base URL of fastasfuck_server")
    parser.add_argument("--trials", type=int, default=3, help="Number of measurement trials per profile (default: 3)")
    parser.add_argument("--output", default="benchmark_mixed_concurrency.json", help="Path to save JSON report")
    args = parser.parse_args()

    run_mixed_suite(base_url=args.url, n_trials=args.trials, output_json=args.output)
