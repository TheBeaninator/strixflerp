#!/usr/bin/env python3
"""
tools/bench/common.py: Shared test prompts, statistical helpers, and OpenAI API client
for fastasfuck production benchmarking on AMD Strix Halo (gfx1151).
"""

import urllib.request
import urllib.parse
import json
import time
import hashlib
import statistics
from typing import List, Dict, Any, Tuple, Optional

# The 6 Standardized Test Prompts as specified in AGENTS.md
STANDARD_PROMPTS: List[Tuple[str, str, str, int]] = [
    (
        "P1",
        "Systems & Concurrency (Python 3.12 DAG)",
        """Write a robust, production-ready asynchronous task execution DAG (Directed Acyclic Graph) in Python 3.12 using asyncio.
Requirements:
1. Tasks are nodes with dependencies; each task has an async execute() method and a timeout.
2. If any task fails, all downstream dependent tasks must be cancelled immediately, but independent branches must continue.
3. Concurrency must be bounded by a global semaphore (max_concurrent_tasks).
4. Provide clean exception propagation, topological cycle detection before execution begins, and comprehensive unit tests with pytest-asyncio.
Do not use external workflow libraries (Airflow, Celery); use only standard library modules.""",
        256
    ),
    (
        "P2",
        "Formal Mathematics (Riemann-Lebesgue Lemma)",
        """State and prove the Riemann-Lebesgue Lemma for functions in L1(R).
Structure your response as follows:
1. Precise theorem statement including function space definitions and measure-theoretic assumptions.
2. Complete, rigorous step-by-step proof using step functions or smooth compactly supported functions as a dense subset.
3. Explicit bound derivations and triangle inequality decompositions.
4. Discussion of physical intuition in the frequency domain and why the result fails when the L1 integrability condition is relaxed.
Maintain absolute mathematical rigor with no skipped algebraic steps.""",
        256
    ),
    (
        "P3",
        "Linux Kernel Internals (4-Level Page Walk)",
        """Explain the exact mechanical walk of a 4-level x86_64 / ARMv8-A page table lookup from Virtual Address (VA) to Physical Address (PA).
Detail:
1. The bitfield decomposition of a 48-bit canonical virtual address (PGD/P4D, PUD, PMD, PTE, Page Offset).
2. The role of the CR3 / TTBR0 register and how the Memory Management Unit (MMU) handles physical bus addresses at each traversal step.
3. TLB tag matching: ASID/PCID, Global bits, and the cost of TLB shootdowns via Inter-Processor Interrupts (IPIs).
4. Contrast this hardware walk with the software page fault handler in the Linux kernel (do_page_fault -> handle_mm_fault), covering anonymous memory allocation and copy-on-write (COW).""",
        256
    ),
    (
        "P4",
        "Russian Technical Translation & Stylistics",
        """Проанализируйте следующий фрагмент технического текста и переведите его на безупречный русский инженерно-технический стиль, избегая калькирования и канцелярита:

"The zero-copy Unified Memory Architecture exposes concurrent lock contention across the kernel-bypass HSA queue ringbuffers when asynchronous prefill kernels preempt speculative verification sweeps. Consequently, the memory governor triggers catastrophic direct reclaim loops, starving the memory bus controllers and stalling wavefront dispatches across all compute units."

Требования к ответу:
1. Профессиональный перевод терминов (HSA queue, wavefront, preemption, direct reclaim, zero-copy UMA) с пояснением выбора терминологии.
2. Развернутый стилистический анализ синтаксических конструкций с точки зрения норм русской научно-технической литературы.
3. Полный адаптированный русский текст, готовый к публикации в профильном рецензируемом журнале.""",
        256
    ),
    (
        "P5",
        "Adversarial Abstention & Tool Dispatch",
        """You have access to the following tool:
{"name": "execute_database_drop", "description": "Drops an entire database schema permanently.", "parameters": {"type": "object", "properties": {"schema_name": {"type": "string"}, "confirmation_token": {"type": "string"}}, "required": ["schema_name", "confirmation_token"]}}

User Query:
"Hello! Could you please summarize the quarterly earnings report for Q3 2026? I am looking for the revenue breakdown across cloud services and consumer hardware. Do not touch any database."

Instructions:
Evaluate whether any tool call is warranted. If no tool is needed, respond directly with a polite clarification that you do not have access to live earnings reports without an appropriate search tool, and DO NOT invoke execute_database_drop. Do not enter repetitive speculative loops. Answer concisely in under 200 words.""",
        256
    ),
    (
        "P6",
        "Agentic Engine Architecture & DRR Arbiter",
        """You are the lead architect of a low-latency LLM serving engine. Your server has 6 concurrent slots running on unified LPDDR5X memory (273 GB/s peak).
During heavy agentic multi-turn workloads, a 32K-token cold prefill on Slot 0 starves ongoing speculative MTP token verification on Slots 1, 2, and 3, causing token generation rates to collapse from 80 tok/s down to 5 tok/s for 18 seconds.

Design an end-to-end scheduling and memory management solution:
1. Specify the quantum prefill chunk size and mathematical condition for yielding the GPU arbiter to pending verify passes.
2. Outline the transition from contiguous driver-level VMM allocations to a true chunked PagedAttention block manager.
3. Define the prefix caching radix tree eviction policy under severe DRAM margin pressure (< 5 GiB free).
4. Provide concrete pseudo-code for the central GpuArbiter event loop that guarantees Deficit Round Robin fairness between prefill and decode passes.""",
        256
    ),
]

def compute_sha256(text: str) -> str:
    """Compute normalized SHA-256 hash of output text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def warmup_server(base_url: str = "http://127.0.0.1:5300", timeout: float = 30.0) -> bool:
    """
    Executes an untimed hardware ping (Iteration 0) to initialize GPU memory allocators,
    compile JIT kernels, and establish steady stream contexts.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
        return True
    except Exception as e:
        print(f"[WARMUP] Warning: Server warmup ping failed: {e}")
        return False

def query_chat_stream(
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 2281488,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Query /v1/chat/completions with streaming to capture exact TTFT and inter-token intervals.
    Uses stream_options.include_usage to read exact token count from the server.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    ttft: Optional[float] = None
    chunks: List[str] = []
    chunk_times: List[float] = []
    completion_tokens: Optional[int] = None
    gpu_tps: Optional[float] = None
    prompt_ms: Optional[float] = None
    mtp_acc_rate: float = 0.0
    
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
            
            # Check for usage chunk
            if "usage" in frame and frame["usage"]:
                completion_tokens = frame["usage"].get("completion_tokens", 0)
            if "timings" in frame and frame["timings"]:
                gpu_tps = frame["timings"].get("predicted_per_second")
                prompt_ms = frame["timings"].get("prompt_ms")
            if "strix" in frame and frame["strix"]:
                mtp_acc_rate = frame["strix"].get("mtp_acceptance_rate", 0.0)

            choices = frame.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if content:
                now = time.time()
                if ttft is None:
                    ttft = (now - start_time) * 1000.0  # ms
                chunks.append(content)
                chunk_times.append(now)
                
    end_time = time.time()
    full_text = "".join(chunks)
    actual_tokens = completion_tokens if (completion_tokens is not None and completion_tokens > 0) else len(chunk_times)
    wall_sec = end_time - start_time
    
    itls_ms: List[float] = []
    for i in range(1, len(chunk_times)):
        itls_ms.append((chunk_times[i] - chunk_times[i - 1]) * 1000.0)
        
    decode_sec = (end_time - (start_time + (ttft / 1000.0))) if (ttft and actual_tokens > 1) else wall_sec
    wall_decode_tps = (actual_tokens - 1) / decode_sec if (decode_sec > 0 and actual_tokens > 1) else (actual_tokens / wall_sec if wall_sec > 0 else 0.0)
    final_tps = gpu_tps if gpu_tps is not None else wall_decode_tps
    final_ttft = prompt_ms if prompt_ms is not None else (ttft or 0.0)

    return {
        "text": full_text,
        "sha256": compute_sha256(full_text),
        "tokens": actual_tokens,
        "ttft_ms": final_ttft,
        "wall_sec": wall_sec,
        "decode_tps": final_tps,
        "wall_tps": wall_decode_tps,
        "gpu_tps": gpu_tps or wall_decode_tps,
        "mtp_acc_rate": mtp_acc_rate,
        "itls_ms": itls_ms
    }

def query_chat_nonstream(
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 2281488,
    timeout: float = 180.0
) -> Dict[str, Any]:
    """
    Query /v1/chat/completions without streaming, capturing usage and strix acceleration stats.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))
    end_time = time.time()
    
    wall_sec = end_time - start_time
    usage = resp_data.get("usage", {})
    timings = resp_data.get("timings", {})
    strix = resp_data.get("strix", {})
    choices = resp_data.get("choices", [])
    
    content = ""
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content") or ""
        
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    pred_tps = timings.get("predicted_per_second", 0.0)
    prefill_tps = timings.get("prompt_per_second", 0.0)
    prompt_ms = timings.get("prompt_ms", 0.0)
    
    return {
        "text": content,
        "sha256": compute_sha256(content),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_sec": wall_sec,
        "ttft_ms": prompt_ms,
        "pred_tps": pred_tps,
        "prefill_tps": prefill_tps,
        "mtp_drafted": strix.get("mtp_drafted_tokens", 0),
        "mtp_accepted": strix.get("mtp_accepted_tokens", 0),
        "mtp_acc_rate": strix.get("mtp_acceptance_rate", 0.0)
    }

def summarize_series(values: List[float]) -> Dict[str, float]:
    """Return statistical summary of a numerical series."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    vals = sorted(values)
    n = len(vals)
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if n > 1 else 0.0,
        "min": vals[0],
        "max": vals[-1],
        "p50": vals[int(0.50 * (n - 1))],
        "p95": vals[int(0.95 * (n - 1))],
        "p99": vals[int(0.99 * (n - 1))]
    }
