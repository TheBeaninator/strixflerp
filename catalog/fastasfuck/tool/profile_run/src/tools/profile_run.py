#!/usr/bin/env python3
"""
Profiles real Hybrid Q6 weights (weights-q6-hybrid.faf) on AMD Strix Halo using rocprofv3 with ROCTX tracing.
1. Stops production server on 5300.
2. Launches strix-roctx server under rocprofv3 on port 5300.
3. Sends warmup ping, followed by queries targeting T = 1, T = 4, T = 16, T = 512, plus speculative generation.
4. Gracefully terminates server (SIGTERM) to flush trace database.
5. Restores production server and verifies health.
6. Analyzes SQLite database for both ROCTX regions (layers, stages, verifies, drafts) and low-level GPU kernels.
"""

import os
import sys
import time
import glob
import json
import signal
import sqlite3
import subprocess
import urllib.request

ROCPROF = "/opt/rocm/bin/rocprofv3"
TRACE_DIR = "/tmp/strix_trace"
EXP_BIN = "/home/cheblan/fastasfuck/build/strix-roctx/fastasfuck_server"
EXP_CONF = "/home/cheblan/strix-server-hybrid.conf"
PROD_SCRIPT = "/home/cheblan/run_thinkingcap.sh"

def run_cmd(cmd, check=True):
    print(f"[CMD] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd}")
    return res

def wait_for_health(port=5300, timeout_s=150):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    print(f"Waiting for {url} to become healthy (timeout: {timeout_s}s)...", flush=True)
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        print(f"Server on port {port} is healthy!", flush=True)
                        return True
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for port {port}")

def send_chat(prompt, max_tokens=1, temp=0.0, port=5300):
    payload = {
        "model": "thinkingcap-qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0
    toks = res.get("usage", {}).get("completion_tokens", 0)
    prompt_toks = res.get("usage", {}).get("prompt_tokens", 0)
    return prompt_toks, toks, elapsed, res

def main():
    proc = None
    try:
        print("\n>>> STEP 1: Stopping Production Server on 5300...", flush=True)
        run_cmd(f"{PROD_SCRIPT} stop", check=False)
        time.sleep(3)

        # Clean trace dir
        run_cmd(f"rm -rf {TRACE_DIR} && mkdir -p {TRACE_DIR}")

        print("\n>>> STEP 2: Launching Server under rocprofv3 on 5300...", flush=True)
        cmd = f"{ROCPROF} --kernel-trace --marker-trace -f rocpd -d {TRACE_DIR} -- {EXP_BIN} --config {EXP_CONF} </dev/null > /tmp/rocprof_server.log 2>&1"
        print(f"[CMD] {cmd}", flush=True)
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

        wait_for_health(5300, timeout_s=150)

        print("\n>>> STEP 3: Executing Target Profiles (T = 1, 4, 16, 512 and MTP Draft)...", flush=True)
        # Warmup ping
        p_tok, g_tok, el, _ = send_chat("ping", max_tokens=5, temp=0.0)
        print(f"  [Warmup] prompt={p_tok}, gen={g_tok} in {el:.2f}s", flush=True)

        # T = 1: Single-token generation from short prompt
        p_tok, g_tok, el, _ = send_chat("What is 2+2?", max_tokens=1, temp=0.0)
        print(f"  [T=1 test] prompt={p_tok}, gen={g_tok} in {el*1000:.1f}ms", flush=True)

        # T = 4: Prompt of ~4 tokens
        p_tok, g_tok, el, _ = send_chat("cat dog apple sun", max_tokens=1, temp=0.0)
        print(f"  [T=4 test] prompt={p_tok}, gen={g_tok} in {el*1000:.1f}ms", flush=True)

        # T = 16: Prompt of ~16 tokens
        p16 = "The rapid development of unified memory architectures has transformed modern hardware-assisted LLM inference engines worldwide today."
        p_tok, g_tok, el, _ = send_chat(p16, max_tokens=1, temp=0.0)
        print(f"  [T=16 test] prompt={p_tok}, gen={g_tok} in {el*1000:.1f}ms", flush=True)

        # T = 512: Prompt of ~512 tokens
        p512 = ("Explain in deep technical detail the mechanical walk of virtual address translation "
                "across four-level page tables on x86_64 architecture with CR3, PGD, PUD, PMD, and PTE registers. ") * 16
        p_tok, g_tok, el, _ = send_chat(p512, max_tokens=1, temp=0.0)
        print(f"  [T=512 prefill test] prompt={p_tok}, gen={g_tok} in {el*1000:.1f}ms", flush=True)

        # Speculative run: 50 tokens generated to record verify and draft ROCTX ranges
        p_spec = "Write a concise C++20 coroutine generator using std::suspend_always."
        p_tok, g_tok, el, res = send_chat(p_spec, max_tokens=50, temp=0.0)
        strix = res.get("strix", {})
        print(f"  [Speculative run] generated={g_tok} in {el:.2f}s ({g_tok/el:.2f} tok/s), MTP: {strix.get('mtp_accepted_tokens', 0)}/{strix.get('mtp_drafted_tokens', 0)} accepted", flush=True)

        print("\n>>> STEP 4: Stopping Server Gracefully to flush trace database...", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            print(f"killpg error: {e}", flush=True)

        try:
            proc.wait(timeout=45)
            print("Server process finalized cleanly.", flush=True)
        except subprocess.TimeoutExpired:
            print("Server did not stop within 45s, sending SIGKILL...", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(3)

    finally:
        print("\n>>> STEP 5: Restoring Production Server on 5300...", flush=True)
        subprocess.run("pkill -9 -f fastasfuck_server || true", shell=True)
        time.sleep(2)
        run_cmd(f"{PROD_SCRIPT} start")
        try:
            wait_for_health(5300, timeout_s=150)
            print("Production server healthy.", flush=True)
        except Exception as e:
            print(f"Error restoring prod: {e}", flush=True)

    # Step 6: Parse SQLite DB
    print("\n" + "=" * 92, flush=True)
    print("                      ROCPROF REAL RUN ROCTX PROFILE ANALYSIS                           ")
    print("=" * 92, flush=True)

    db_files = glob.glob(f"{TRACE_DIR}/**/*.db", recursive=True)
    if not db_files:
        print(f"[ERROR] No .db file found in {TRACE_DIR}", flush=True)
        run_cmd(f"ls -laR {TRACE_DIR}", check=False)
        return

    db_path = db_files[0]
    print(f"Reading trace database: {db_path}", flush=True)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # 1. Query ROCTX Regions (Verifies, Prefills, Drafts, Stages)
    print("\n--- TOP ROCTX RANGES (VERIFY / PREFILL / DRAFT / LAYERS) ---", flush=True)
    try:
        cur = con.execute("""
            SELECT name, count(*), avg(duration)/1e6, sum(duration)/1e6
            FROM regions
            GROUP BY name
            ORDER BY sum(duration) DESC
            LIMIT 40
        """)
        rows = cur.fetchall()
        print(f"{'ROCTX Marker / Range Name':<50} | {'Calls':>6} | {'Avg (ms)':>10} | {'Total (ms)':>10}", flush=True)
        print("-" * 84, flush=True)
        for name, cnt, avg_ms, tot_ms in rows:
            print(f"{name[:50]:<50} | {cnt:>6} | {avg_ms:>10.3f} | {tot_ms:>10.2f}", flush=True)
    except Exception as e:
        print(f"Could not query regions: {e}", flush=True)

    # 2. Query GPU Kernels
    print("\n--- TOP GPU KERNEL DISPATCHES ---", flush=True)
    try:
        cur = con.execute("""
            SELECT name, count(*), avg(duration)/1e3, sum(duration)/1e6
            FROM kernels
            GROUP BY name
            ORDER BY sum(duration) DESC
            LIMIT 30
        """)
        rows = cur.fetchall()
        tot_time = sum(r[3] for r in rows)
        print(f"{'Kernel Symbol Name':<50} | {'Calls':>6} | {'Avg (us)':>10} | {'Total (ms)':>10} | {'GPU %':>6}", flush=True)
        print("-" * 92, flush=True)
        for name, cnt, avg_us, tot_ms in rows:
            short_name = name.split("(")[0].split("<")[0]
            pct = (tot_ms / tot_time * 100.0) if tot_time > 0 else 0
            print(f"{short_name[:50]:<50} | {cnt:>6} | {avg_us:>10.2f} | {tot_ms:>10.2f} | {pct:>5.1f}%", flush=True)
    except Exception as e:
        print(f"Could not query kernels: {e}", flush=True)

    con.close()
    print("=" * 92, flush=True)

if __name__ == "__main__":
    main()
