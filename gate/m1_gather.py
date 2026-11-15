#!/usr/bin/env python3
"""M1 of docs/PHASE5-FOURNODE-DESIGN.md section 6: one eager all-gather of BYTES between two nodes, timed. Each
exchange: both ranks send their BYTES and receive the peer's at the same time (an all-gather at world 2) over one TCP
connection (TCP_NODELAY, the transport runtime.flerp.stage_relay uses), host memory only. Stdlib only.

    rank 1:  python3 m1_gather.py --listen 198.51.100.53 --port 39790
    rank 0:  python3 m1_gather.py --connect 198.51.100.53 --port 39790 [--bytes 163840 --reps 300 --warm 30]

Rank 0 prints one JSON object: per exchange milliseconds (after the warm-up), median, p10, p90, min, max.
"""
import argparse
import json
import select
import socket
import statistics
import time


def exchange(s, out, n):
    """Send OUT and receive N bytes concurrently; returns the received bytes."""
    sent, got = 0, bytearray()
    while sent < len(out) or len(got) < n:
        w = [s] if sent < len(out) else []
        r, wr, _ = select.select([s] if len(got) < n else [], w, [], 30)
        if not r and not wr:
            raise TimeoutError("exchange stalled 30 s")
        if wr:
            try:
                sent += s.send(out[sent:sent + (1 << 20)])
            except BlockingIOError:
                pass
        if r:
            try:
                b = s.recv(n - len(got))
            except BlockingIOError:
                b = None
            if b == b"":
                raise ConnectionError("peer closed")
            if b:
                got += b
    return got


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--listen", metavar="ADDR")
    g.add_argument("--connect", metavar="ADDR")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bytes", type=int, default=160 * 1024)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warm", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()
    if a.listen:
        ls = socket.socket()
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((a.listen, a.port))
        ls.listen(1)
        ls.settimeout(a.timeout)
        s, _ = ls.accept()
        ls.close()
    else:
        end = time.time() + a.timeout
        while True:
            try:
                s = socket.create_connection((a.connect, a.port), timeout=5)
                break
            except OSError:
                if time.time() > end:
                    raise
                time.sleep(0.2)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setblocking(False)
    out = bytes((i * 7 + (1 if a.listen else 0)) & 255 for i in range(a.bytes))
    times = []
    for i in range(a.warm + a.reps):
        t0 = time.perf_counter()
        got = exchange(s, out, a.bytes)
        dt = (time.perf_counter() - t0) * 1e3
        if len(got) != a.bytes:
            raise RuntimeError("short exchange")
        if i >= a.warm:
            times.append(dt)
    s.close()
    if a.connect:
        q = statistics.quantiles(times, n=10)
        print(json.dumps({"event": "m1", "bytes": a.bytes, "reps": a.reps, "warm": a.warm, "peer": a.connect,
                          "median_ms": statistics.median(times), "p10_ms": q[0], "p90_ms": q[-1], "min_ms": min(times),
                          "max_ms": max(times), "mean_ms": statistics.fmean(times), "ms": [round(t, 4) for t in times]}))


if __name__ == "__main__":
    main()
