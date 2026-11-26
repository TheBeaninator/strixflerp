#!/usr/bin/env python3
"""
tools/bench/bench_ab_quant.py: Controlled AGENTS.md-compliant A/B benchmark suite
comparing two quantization layouts on the same model weights & server config.

Follows AGENTS.md Section 4:
  - --prompt-cache off (verified via server cache_n == 0)
  - Untimed hardware warmup (>=50 prompt & >=50 gen tokens, distinct from P1..P6)
  - Solo sterile (n=5 across P1..P6, T=0.0, dynamic SHA-256 bit-identity check)
  - Live MTP acceptance rate (solo & 4-slot concurrent)
  - 4-Slot Concurrency with MTP (1 warmup + n=3 trials across 4 concurrent slots, T=0.6)
  - 4k & 8k Short Prefill Scaling (non-repeating C++ corpus, 1 cold pass + n=5 warm passes)
"""

import argparse
import concurrent.futures
import glob
import json
import os
import statistics
import sys
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STANDARD_PROMPTS, compute_sha256, summarize_series


WARMUP_LONG_PROMPT = (
    "Briefly explain why Unified Memory Architecture (UMA) on AMD Strix Halo couples CPU "
    "direct memory reclaim pressure directly to GPU wavefront dispatch stalls over shared "
    "LPDDR5X-8533 channels, and list three kernel parameters that mitigate page thrashing "
    "under multi-slot LLM inference workloads."
)


def query_stream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 2281488,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Streaming chat completion capturing timings, usage, strix MTP stats, ITLs, and SHA-256."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft_client_ms: Optional[float] = None
    chunks: List[str] = []
    chunk_times: List[float] = []
    usage: Dict[str, Any] = {}
    timings: Dict[str, Any] = {}
    strix: Dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                frame = json.loads(data_str)
            except Exception:
                continue

            if frame.get("usage"):
                usage = frame["usage"]
            if frame.get("timings"):
                timings = frame["timings"]
            if frame.get("strix"):
                strix = frame["strix"]

            choices = frame.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                now = time.time()
                if ttft_client_ms is None:
                    ttft_client_ms = (now - t0) * 1000.0
                chunks.append(piece)
                chunk_times.append(now)

    wall_sec = time.time() - t0
    full_text = "".join(chunks)
    comp_tokens = int(usage.get("completion_tokens", len(chunk_times)) or len(chunk_times))
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    cache_n = int(timings.get("cache_n", 0) or 0)
    prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
    pred_tps = float(timings.get("predicted_per_second", 0.0) or 0.0)
    prefill_tps = float(timings.get("prompt_per_second", 0.0) or 0.0)

    itls_ms = [(chunk_times[i] - chunk_times[i - 1]) * 1000.0 for i in range(1, len(chunk_times))]
    if pred_tps <= 0.0 and wall_sec > 0.0 and comp_tokens > 1:
        dec_s = wall_sec - ((ttft_client_ms or 0.0) / 1000.0)
        pred_tps = (comp_tokens - 1) / dec_s if dec_s > 0.0 else comp_tokens / wall_sec

    return {
        "text": full_text,
        "sha256": compute_sha256(full_text),
        "prompt_tokens": prompt_tokens,
        "tokens": comp_tokens,
        "cache_n": cache_n,
        "ttft_ms": prompt_ms if prompt_ms > 0.0 else (ttft_client_ms or 0.0),
        "wall_sec": wall_sec,
        "decode_tps": pred_tps,
        "prefill_tps": prefill_tps,
        "mtp_drafted": int(strix.get("mtp_drafted_tokens", 0) or 0),
        "mtp_accepted": int(strix.get("mtp_accepted_tokens", 0) or 0),
        "mtp_acc_rate": float(strix.get("mtp_acceptance_rate", 0.0) or 0.0),
        "mtp_draft_calls": int(strix.get("mtp_draft_calls", 0) or 0),
        "mtp_avg_draft_ms": float(strix.get("mtp_avg_draft_ms", 0.0) or 0.0),
        "itls_ms": itls_ms,
    }


def query_nonstream(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 5,
    temperature: float = 0.0,
    seed: int = 2281488,
    timeout: float = 240.0,
) -> Dict[str, Any]:
    """Non-streaming completion used for prefill scaling and warmup."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    wall_sec = time.time() - t0
    usage = data.get("usage", {})
    timings = data.get("timings", {})
    strix = data.get("strix", {})
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "prompt_n": int(timings.get("prompt_n", 0) or 0),
        "cache_n": int(timings.get("cache_n", 0) or 0),
        "prompt_ms": float(timings.get("prompt_ms", 0.0) or 0.0),
        "prefill_tps": float(timings.get("prompt_per_second", 0.0) or 0.0),
        "pred_tps": float(timings.get("predicted_per_second", 0.0) or 0.0),
        "wall_sec": wall_sec,
        "mtp_acc_rate": float(strix.get("mtp_acceptance_rate", 0.0) or 0.0),
    }


def flush_slots(base_url: str, model: str, n_slots: int = 4) -> None:
    """Send n_slots concurrent 1-token unique requests to clear live in-slot KV state for sterile runs."""
    nonce = uuid.uuid4().hex[:8]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as ex:
        futs = [
            ex.submit(
                query_nonstream,
                base_url,
                model,
                f"reset slot {i} {nonce}",
                1,
                0.0,
                2281488 + i,
                60.0,
            )
            for i in range(n_slots)
        ]
        for f in futs:
            _ = f.result()


def run_untimed_warmup(base_url: str, model: str) -> None:
    print("\n[WARMUP] Ping + 64-tok untimed warmup (distinct from P1..P6)...")
    r1 = query_nonstream(base_url, model, "ping", max_tokens=5, temperature=0.0, timeout=60.0)
    r2 = query_nonstream(base_url, model, WARMUP_LONG_PROMPT, max_tokens=64, temperature=0.0, timeout=120.0)
    print(
        f"[WARMUP] Ping: {r1['prompt_tokens']}p/{r1['completion_tokens']}g | "
        f"Turn: {r2['prompt_tokens']}p/{r2['completion_tokens']}g ({r2['pred_tps']:.2f} tok/s, cache_n={r2['cache_n']}) [discarded]\n"
    )


def run_solo_sterile(base_url: str, model: str, n_trials: int = 5) -> Dict[str, Any]:
    print("=" * 96)
    print(f"SOLO STERILE BENCHMARK (1 Slot, T=0.0, n={n_trials} trials across P1..P6, trial-outer sterile)")
    print("=" * 96)

    per_pid: Dict[str, Dict[str, Any]] = {}
    for pid, domain, _, _ in STANDARD_PROMPTS:
        per_pid[pid] = {
            "domain": domain,
            "tpss": [],
            "ttfts": [],
            "walls": [],
            "draft_mss": [],
            "itls": [],
            "hashes": [],
            "drafted": 0,
            "accepted": 0,
            "tokens": 0,
        }

    nonzero_cache = 0
    for trial in range(1, n_trials + 1):
        flush_slots(base_url, model, n_slots=4)
        for pid, domain, prompt, max_tokens in STANDARD_PROMPTS:
            r = query_stream(base_url, model, prompt, max_tokens=max_tokens, temperature=0.0, seed=2281488)
            if r["cache_n"] > 0:
                nonzero_cache += 1
            st = per_pid[pid]
            if trial == 1:
                st["tokens"] = r["tokens"]
            st["tpss"].append(r["decode_tps"])
            st["ttfts"].append(r["ttft_ms"])
            st["walls"].append(r["wall_sec"])
            if r["mtp_draft_calls"] > 0:
                st["draft_mss"].append(r["mtp_avg_draft_ms"])
            st["itls"].extend(r["itls_ms"])
            st["hashes"].append(r["sha256"])
            st["drafted"] += r["mtp_drafted"]
            st["accepted"] += r["mtp_accepted"]
            acc_str = f" | acc={r['mtp_acc_rate']:5.1f}% ({r['mtp_accepted']}/{r['mtp_drafted']})" if r["mtp_drafted"] > 0 else ""
            print(
                f"  T{trial}/{n_trials} {pid}: {r['tokens']:3d} tok | Decode={r['decode_tps']:6.2f} tok/s | "
                f"TTFT={r['ttft_ms']:6.1f} ms | cache_n={r['cache_n']}{acc_str} | sha={r['sha256'][:10]}"
            )

    tasks: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    tot_drafted = 0
    tot_accepted = 0

    for pid, domain, _, _ in STANDARD_PROMPTS:
        st = per_pid[pid]
        hashes = st["hashes"]
        all_eq = all(h == hashes[0] for h in hashes)
        p_drafted = st["drafted"]
        p_accepted = st["accepted"]
        tot_drafted += p_drafted
        tot_accepted += p_accepted
        p_acc = (100.0 * p_accepted / p_drafted) if p_drafted > 0 else 0.0
        s_tps = summarize_series(st["tpss"])
        s_ttft = summarize_series(st["ttfts"])
        s_wall = summarize_series(st["walls"])
        s_itl = summarize_series(st["itls"])
        mean_draft_ms = statistics.mean(st["draft_mss"]) if st["draft_mss"] else 0.0

        tasks[pid] = {
            "domain": domain,
            "tokens": st["tokens"],
            "decode_tps": s_tps,
            "ttft_ms": s_ttft,
            "wall_sec": s_wall,
            "itl_ms": s_itl,
            "mtp_drafted": p_drafted,
            "mtp_accepted": p_accepted,
            "mtp_acc_rate_pct": p_acc,
            "mtp_avg_draft_ms": mean_draft_ms,
            "deterministic": all_eq,
            "sha256": hashes[0],
        }
        rows.append({
            "pid": pid,
            "tps_mean": s_tps["mean"],
            "tps_std": s_tps["std"],
            "ttft_mean": s_ttft["mean"],
            "itl_p50": s_itl["p50"],
            "itl_p95": s_itl["p95"],
            "itl_p99": s_itl["p99"],
            "draft_ms": mean_draft_ms,
            "acc_pct": p_acc,
            "det": all_eq,
        })

    grand_tps = statistics.mean(r["tps_mean"] for r in rows)
    grand_tps_std = statistics.stdev(r["tps_mean"] for r in rows)
    grand_ttft = statistics.mean(r["ttft_mean"] for r in rows)
    grand_itl_p50 = statistics.mean(r["itl_p50"] for r in rows)
    grand_itl_p95 = statistics.mean(r["itl_p95"] for r in rows)
    grand_itl_p99 = statistics.mean(r["itl_p99"] for r in rows)
    active_draft_ms = [r["draft_ms"] for r in rows if r["draft_ms"] > 0.0]
    mean_mtp_draft_ms = statistics.mean(active_draft_ms) if active_draft_ms else 0.0
    global_acc = (100.0 * tot_accepted / tot_drafted) if tot_drafted > 0 else 0.0
    all_det = all(r["det"] for r in rows)

    print("-" * 96)
    print(f"{'Task':4s} | {'Decode tok/s':15s} | {'TTFT (ms)':10s} | {'ITL p50/p95':13s} | {'MTP Acc%':9s} | {'Bit-Identity'}")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['pid']:4s} | {r['tps_mean']:6.2f} ± {r['tps_std']:4.2f}  | {r['ttft_mean']:8.1f}  | "
            f"{r['itl_p50']:5.1f} / {r['itl_p95']:5.1f} | {r['acc_pct']:7.1f}%  | {'100% MATCH' if r['det'] else 'FAIL'}"
        )
    print("-" * 96)
    print(
        f"GRAND MEAN: {grand_tps:.2f} tok/s (across-prompt std {grand_tps_std:.2f}) | "
        f"TTFT={grand_ttft:.1f} ms | ITL p50/p95/p99={grand_itl_p50:.1f}/{grand_itl_p95:.1f}/{grand_itl_p99:.1f} ms | "
        f"Global MTP Acc={global_acc:.2f}% ({tot_accepted}/{tot_drafted}, draft={mean_mtp_draft_ms:.2f}ms) | "
        f"Bit-ID={'PASS' if all_det else 'FAIL'} | nonzero_cache={nonzero_cache}"
    )

    return {
        "n_trials": n_trials,
        "grand_mean_decode_tps": grand_tps,
        "across_prompt_std_tps": grand_tps_std,
        "grand_mean_ttft_ms": grand_ttft,
        "grand_mean_itl_p50_ms": grand_itl_p50,
        "grand_mean_itl_p95_ms": grand_itl_p95,
        "grand_mean_itl_p99_ms": grand_itl_p99,
        "mean_mtp_avg_draft_ms": mean_mtp_draft_ms,
        "global_mtp_drafted": tot_drafted,
        "global_mtp_accepted": tot_accepted,
        "global_mtp_acc_rate_pct": global_acc,
        "all_deterministic": all_det,
        "nonzero_cache_count": nonzero_cache,
        "tasks": tasks,
    }


def run_multislot_4(base_url: str, model: str, n_trials: int = 3, n_slots: int = 4) -> Dict[str, Any]:
    print("\n" + "=" * 96)
    print(f"MULTI-SLOT CONCURRENCY BENCHMARK ({n_slots} Active Slots with MTP, T=0.6, n={n_trials} + warmup)")
    print("=" * 96)

    def one_pass(prompt_indices: List[int], seed_base: int) -> Dict[str, Any]:
        prompts = [STANDARD_PROMPTS[idx % len(STANDARD_PROMPTS)] for idx in prompt_indices]
        t0 = time.time()
        res_list: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as ex:
            futs = [
                ex.submit(query_stream, base_url, model, p[2], p[3], 0.6, seed_base + i)
                for i, p in enumerate(prompts)
            ]
            for f in concurrent.futures.as_completed(futs):
                res_list.append(f.result())
        wall = time.time() - t0
        tot_tok = sum(r["tokens"] for r in res_list)
        drafted = sum(r["mtp_drafted"] for r in res_list)
        accepted = sum(r["mtp_accepted"] for r in res_list)
        itls: List[float] = []
        for r in res_list:
            itls.extend(r["itls_ms"])
        return {
            "wall_sec": wall,
            "total_tokens": tot_tok,
            "agg_tps": tot_tok / wall if wall > 0 else 0.0,
            "slot_tps_mean": statistics.mean(r["decode_tps"] for r in res_list),
            "drafted": drafted,
            "accepted": accepted,
            "acc_pct": (100.0 * accepted / drafted) if drafted > 0 else 0.0,
            "itl_stats": summarize_series(itls),
            "cache_hits": sum(1 for r in res_list if r["cache_n"] > 0),
        }

    flush_slots(base_url, model, n_slots=n_slots)
    _ = one_pass([0, 1, 2, 3], 111000)
    print("  Warmup pass complete [discarded].")

    aggs: List[float] = []
    slots: List[float] = []
    p50s: List[float] = []
    p95s: List[float] = []
    p99s: List[float] = []
    tot_drafted = 0
    tot_accepted = 0
    tot_cache_hits = 0
    trials_out: List[Dict[str, Any]] = []

    for t in range(1, n_trials + 1):
        flush_slots(base_url, model, n_slots=n_slots)
        idxs = [(t - 1 + s) % len(STANDARD_PROMPTS) for s in range(n_slots)]
        pr = one_pass(idxs, 2281488 + t * 100)
        aggs.append(pr["agg_tps"])
        slots.append(pr["slot_tps_mean"])
        p50s.append(pr["itl_stats"]["p50"])
        p95s.append(pr["itl_stats"]["p95"])
        p99s.append(pr["itl_stats"]["p99"])
        tot_drafted += pr["drafted"]
        tot_accepted += pr["accepted"]
        tot_cache_hits += pr["cache_hits"]
        trials_out.append(pr)
        print(
            f"  Trial {t}/{n_trials}: Agg={pr['agg_tps']:6.2f} tok/s | Per-Slot={pr['slot_tps_mean']:5.2f} tok/s | "
            f"ITL p50/p95/p99={pr['itl_stats']['p50']:4.1f}/{pr['itl_stats']['p95']:4.1f}/{pr['itl_stats']['p99']:4.1f} ms | "
            f"Acc={pr['acc_pct']:5.1f}% | cache_hits={pr['cache_hits']} | wall={pr['wall_sec']:.2f}s"
        )

    s_agg = summarize_series(aggs)
    s_slot = summarize_series(slots)
    g_acc = (100.0 * tot_accepted / tot_drafted) if tot_drafted > 0 else 0.0
    print("-" * 96)
    print(
        f"4-SLOT MEAN: Agg={s_agg['mean']:.2f} ± {s_agg['std']:.2f} tok/s | "
        f"Per-Slot={s_slot['mean']:.2f} ± {s_slot['std']:.2f} tok/s | "
        f"ITL p50={statistics.mean(p50s):.1f}ms p95={statistics.mean(p95s):.1f}ms p99={statistics.mean(p99s):.1f}ms | "
        f"Acc={g_acc:.2f}% | cache_hits={tot_cache_hits}"
    )

    return {
        "n_slots": n_slots,
        "n_trials": n_trials,
        "agg_tps": s_agg,
        "slot_tps": s_slot,
        "itl_p50_ms": statistics.mean(p50s),
        "itl_p95_ms": statistics.mean(p95s),
        "itl_p99_ms": statistics.mean(p99s),
        "mtp_drafted": tot_drafted,
        "mtp_accepted": tot_accepted,
        "mtp_acc_rate_pct": g_acc,
        "total_cache_hits": tot_cache_hits,
        "trials": trials_out,
    }


def build_corpus(repo_dir: str) -> str:
    files = sorted(
        glob.glob(os.path.join(repo_dir, "serve", "*.cpp"))
        + glob.glob(os.path.join(repo_dir, "serve", "*.hpp"))
    )
    parts: List[str] = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            parts.append(f.read())
    corpus = "\n\n// NEXT FILE\n\n".join(parts)
    assert len(corpus) > 100000, f"Unexpectedly small corpus at {repo_dir}"
    return corpus


def slice_prompt(corpus: str, target_tokens: int, offset_idx: int, nonce: str) -> str:
    target_chars = int(target_tokens * 3.6)
    start = (offset_idx * 17389) % max(1, len(corpus) - target_chars - 100)
    chunk = corpus[start : start + target_chars]
    return (
        f"// Code Corpus Segment [nonce: {nonce}, target_tokens: {target_tokens}]\n"
        + chunk
        + "\n\n// Summarize the design patterns in one sentence."
    )


def run_prefill_short(base_url: str, model: str, repo_dir: str, n_trials: int = 5) -> Dict[str, Any]:
    print("\n" + "=" * 96)
    print(f"PREFILL SCALING BENCHMARK (4k & 8k non-repeating C++ corpus, 1 cold + n={n_trials} warm)")
    print("=" * 96)
    corpus = build_corpus(repo_dir)
    results: Dict[str, Any] = {}
    offset = 0

    for target_tok, label in [(4096, "4k"), (8192, "8k")]:
        cold_p = slice_prompt(corpus, target_tok, offset, f"cold_{uuid.uuid4().hex[:8]}")
        offset += 1
        rc = query_nonstream(base_url, model, cold_p, max_tokens=5, temperature=0.0)
        nonzero_cache = 1 if rc["cache_n"] > 0 else 0
        print(
            f"  {label} Cold Pass: {rc['prompt_tokens']} prompt ({rc['prompt_n']} evaluated, cache_n={rc['cache_n']}) -> "
            f"{rc['prefill_tps']:7.1f} tok/s ({rc['prompt_ms']:.1f} ms, wall={rc['wall_sec']:.2f}s)"
        )

        warm_tps: List[float] = []
        warm_ms: List[float] = []
        prompt_toks = rc["prompt_tokens"]
        for t in range(1, n_trials + 1):
            wp = slice_prompt(corpus, target_tok, offset, f"warm_{t}_{uuid.uuid4().hex[:8]}")
            rw = query_nonstream(base_url, model, wp, max_tokens=5, temperature=0.0)
            if rw["cache_n"] > 0:
                nonzero_cache += 1
            prompt_toks = rw["prompt_tokens"]
            warm_tps.append(rw["prefill_tps"])
            warm_ms.append(rw["prompt_ms"])
            print(
                f"  {label} T{t}/{n_trials}: {rw['prompt_n']} tok (cache_n={rw['cache_n']}) -> "
                f"{rw['prefill_tps']:7.1f} tok/s ({rw['prompt_ms']:.1f} ms)"
            )
        offset += 1
        s_tps = summarize_series(warm_tps)
        s_ms = summarize_series(warm_ms)
        results[label] = {
            "target_tokens": target_tok,
            "actual_prompt_tokens": prompt_toks,
            "cold_tps": rc["prefill_tps"],
            "cold_prompt_ms": rc["prompt_ms"],
            "warm_tps": s_tps,
            "warm_prompt_ms": s_ms,
            "nonzero_cache_count": nonzero_cache,
        }
        print(f"  => {label} WARM MEAN: {s_tps['mean']:.1f} ± {s_tps['std']:.1f} tok/s ({s_ms['mean']:.1f} ms)\n")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.1.8:5300")
    ap.add_argument("--model", default="thinking-cap")
    ap.add_argument("--suite", choices=["bare", "mtp", "all"], required=True)
    ap.add_argument("--repo", default="/home/cheblan/Desktop/strixite-playground/strixite")
    ap.add_argument("--solo-trials", type=int, default=5)
    ap.add_argument("--multislot-trials", type=int, default=3)
    ap.add_argument("--prefill-trials", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out: Dict[str, Any] = {
        "url": args.url,
        "model": args.model,
        "suite": args.suite,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    run_untimed_warmup(args.url, args.model)

    if args.suite in ("bare", "all"):
        out["solo_bare" if args.suite == "bare" else "solo"] = run_solo_sterile(
            args.url, args.model, n_trials=args.solo_trials
        )

    if args.suite in ("mtp", "all"):
        out["solo_mtp"] = run_solo_sterile(args.url, args.model, n_trials=args.solo_trials)
        out["multislot_4"] = run_multislot_4(args.url, args.model, n_trials=args.multislot_trials, n_slots=4)
        out["prefill_4k_8k"] = run_prefill_short(
            args.url, args.model, repo_dir=args.repo, n_trials=args.prefill_trials
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
