#!/usr/bin/env python3
"""The node-side sampler of gate/pair_stall_diag.py (pod card PAIR-STALL c18e4b0bd2). Python 3 standard library only.
It runs on a worker as a transient user unit and reads nothing but /proc, /sys and the rank's own log and pid file
(which the runner puts on the node's RAM overlay): no NFS path is opened in its loop.

    python3 -I pair_stall_sampler.py --rank 1 --run RUN_DIR --out FILE.jsonl [--interval 2] [--max-seconds 5400]

One JSON object a line in FILE:
  {"kind": "hdr", ...}       host, interfaces with their IPv4 address, NFS mounts (mount point -> device), whether
                             mlockall held this process's pages (so memory pressure cannot page the sampler out)
  {"kind": "s", "t": epoch, "c": {counter: cumulative value}, "g": {gauge: value}}   every --interval seconds
  {"kind": "ev", "t": epoch, "ev": "frame" | "cache" | "pass" | "log", ...}          a line of rank<R>.log, stamped
                             when it appeared (the log has no times of its own); the log is polled every 0.25 s
Counters are cumulative as the kernel gives them; the analysis takes differences.
"""
import argparse
import json
import os
import re
import signal
import socket
import struct
import sys
import time

MEMINFO = ("MemAvailable", "MemFree", "Cached", "Mapped", "Active(file)", "Inactive(file)", "Dirty", "Shmem", "Unevictable")
VMSTAT = ("pgfault", "pgmajfault", "pgscan_kswapd", "pgscan_direct", "pgscan_khugepaged", "pgsteal_kswapd",
          "pgsteal_direct", "pgsteal_khugepaged", "compact_stall", "compact_fail", "compact_success",
          "workingset_refault_file", "workingset_refault_anon", "allocstall_normal", "allocstall_movable", "pgpgin")
BLOCKS = ("loop0", "nvme0n1")     # loop0: the netboot root image (a squashfs read over NFS); nvme0n1: /cache
FRAME = re.compile(r"step (\d+) rank (\d) spent (\d+) ms on its (\w+) frame")
CACHE = re.compile(r"prompt cache: (\d+) tokens, resumed at (\d+), kept (\d+); (\d+) states, (\d+) MiB and (\d+) MiB "
                   r"spare of (\d+) MiB \(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)")
PASS = re.compile(r"step (\d+) rank \d prompt pass: (\d+) tokens")


def parse_psi(text):
    """/proc/pressure/<x> -> {"some_avg10": float, "some_us": int, "full_avg10": ..., "full_us": ...}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
        if "avg10" in kv:
            out[parts[0] + "_avg10"] = float(kv["avg10"])
        if "total" in kv:
            out[parts[0] + "_us"] = int(kv["total"])
    return out


def parse_meminfo(text, keys=MEMINFO):
    """/proc/meminfo -> {key: kB} for `keys`."""
    out = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        if k in keys:
            out[k] = int(rest.split()[0])
    return out


def parse_vmstat(text, keys=VMSTAT):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in keys:
            out[parts[0]] = int(parts[1])
    return out


def parse_mountstats(text):
    """/proc/self/mountstats -> {mount point: {"device", "fstype", "normal_read_bytes", "direct_read_bytes",
    "server_read_bytes", "read_ops", "read_queue_ms", "read_rtt_ms", "read_exe_ms"}} for the NFS mounts.
    bytes: normalread normalwrite directread directwrite serverread serverwrite readpages writepages;
    READ: ops transmissions timeouts bytes_sent bytes_received queue_ms rtt_ms execute_ms [errors]."""
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("device "):
            m = re.match(r"device (\S+) mounted on (\S+) with fstype (\S+)", line)
            cur = None
            if m and m.group(3).startswith("nfs"):
                cur = out.setdefault(m.group(2), {"device": m.group(1), "fstype": m.group(3)})
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("bytes:"):
            n = [int(x) for x in s.split()[1:]]
            if len(n) >= 5:
                cur.update(normal_read_bytes=n[0], direct_read_bytes=n[2], server_read_bytes=n[4])
        elif s.startswith("READ:"):
            n = [int(x) for x in s.split()[1:]]
            if len(n) >= 8:
                cur.update(read_ops=n[0], read_queue_ms=n[5], read_rtt_ms=n[6], read_exe_ms=n[7])
    return out


def parse_netdev(text):
    """/proc/net/dev -> {iface: {"rx_bytes", "tx_bytes"}} without lo."""
    out = {}
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        n = rest.split()
        if sep and len(n) >= 9 and name.strip() != "lo":
            out[name.strip()] = {"rx_bytes": int(n[0]), "tx_bytes": int(n[8])}
    return out


def parse_pid_stat(text):
    """/proc/<pid>/stat -> state, minflt, majflt, utime, stime (ticks), blkio delay (ticks). The command name may
    hold spaces and parentheses: fields are counted after the last ')'."""
    f = text[text.rfind(")") + 2:].split()
    out = {"state": f[0], "minflt": int(f[7]), "majflt": int(f[9]), "utime": int(f[11]), "stime": int(f[12])}
    if len(f) > 39:
        out["blkio_ticks"] = int(f[39])
    return out


def parse_pid_io(text):
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in ("rchar", "read_bytes", "syscr"):
            out[k] = int(v)
    return out


def parse_pid_status(text):
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in ("RssFile", "RssAnon", "RssShmem", "VmRSS"):
            out[k] = int(v.split()[0])
    return out


def parse_block_stat(text):
    """/sys/block/<dev>/stat -> reads completed, sectors read, ms spent reading."""
    n = text.split()
    return {"read_ios": int(n[0]), "read_sectors": int(n[2]), "read_ms": int(n[3])} if len(n) >= 4 else {}


def parse_log_line(line):
    """A line of rank<R>.log -> an event dict without its time, or None."""
    m = FRAME.search(line)
    if m:
        return {"ev": "frame", "step": int(m.group(1)), "rank": int(m.group(2)), "ms": int(m.group(3)), "frame": m.group(4)}
    m = CACHE.search(line)
    if m:
        keys = ("tokens", "resumed", "kept", "states", "mib", "spare_mib", "cap_mib", "hits", "misses", "evicted", "refused", "failed")
        return dict({"ev": "cache"}, **{k: int(v) for k, v in zip(keys, m.groups())})
    m = PASS.search(line)
    if m:
        return {"ev": "pass", "step": int(m.group(1)), "tokens": int(m.group(2))}
    if line.startswith(("warning:", "error:")) or "failed" in line:
        return {"ev": "log", "text": line[:240]}
    return None


def flat_sample(texts):
    """The files' texts of one tick ({name: text}) -> (counters, gauges), flat. Names: psi.<x>, meminfo, vmstat,
    mountstats, netdev, loadavg, stat, blk.<dev>, pid.stat, pid.io, pid.status. A missing text is skipped."""
    c, g = {}, {}
    for res in ("memory", "io", "cpu"):
        for k, v in parse_psi(texts.get("psi." + res, "")).items():
            (c if k.endswith("_us") else g)[f"psi:{res}:{k}"] = v
    for k, v in parse_meminfo(texts.get("meminfo", "")).items():
        g["mem:" + k] = v
    for k, v in parse_vmstat(texts.get("vmstat", "")).items():
        c["vm:" + k] = v
    for mnt, d in parse_mountstats(texts.get("mountstats", "")).items():
        for k, v in d.items():
            if isinstance(v, int):
                c[f"nfs:{mnt}:{k}"] = v
    for ifc, d in parse_netdev(texts.get("netdev", "")).items():
        for k, v in d.items():
            c[f"net:{ifc}:{k}"] = v
    la = texts.get("loadavg", "").split()
    if len(la) >= 4:
        g["load1"] = float(la[0])
        g["runnable"] = int(la[3].split("/")[0])
    for line in texts.get("stat", "").splitlines():
        if line.startswith("procs_blocked"):
            g["procs_blocked"] = int(line.split()[1])
        elif line.startswith("cpu "):
            n = line.split()
            if len(n) > 5:
                c["cpu:iowait_ticks"] = int(n[5])
    for dev in BLOCKS:
        for k, v in parse_block_stat(texts.get("blk." + dev, "")).items():
            c[f"blk:{dev}:{k}"] = v
    if texts.get("pid.stat"):
        d = parse_pid_stat(texts["pid.stat"])
        g["pid:state"] = d.pop("state")
        for k, v in d.items():
            c["pid:" + k] = v
    for k, v in parse_pid_io(texts.get("pid.io", "")).items():
        c["pid:" + k] = v
    for k, v in parse_pid_status(texts.get("pid.status", "")).items():
        g["pid:" + k] = v
    return c, g


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def iface_addrs():
    """{iface: IPv4 address} by ioctl (no subprocess, no file outside /proc)."""
    import fcntl
    out = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for name in parse_netdev(read("/proc/net/dev")):
        try:
            raw = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))   # SIOCGIFADDR
            out[name] = socket.inet_ntoa(raw[20:24])
        except OSError:
            out[name] = None
    s.close()
    return out


def mlock_all():
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.mlockall(3) == 0        # MCL_CURRENT | MCL_FUTURE
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--run", required=True, help="the run directory: rank<R>.pid and rank<R>.log appear in it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=5400, help="the sampler ends by itself after this")
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    out = open(a.out, "a", buffering=1)
    locked = mlock_all()
    out.write(json.dumps({"kind": "hdr", "t": time.time(), "host": socket.gethostname(), "rank": a.rank, "interval": a.interval,
                          "mlocked": locked, "ifaces": iface_addrs(),
                          "mounts": {m: d["device"] for m, d in parse_mountstats(read("/proc/self/mountstats")).items()}}) + "\n")
    pidfile, logfile = os.path.join(a.run, f"rank{a.rank}.pid"), os.path.join(a.run, f"rank{a.rank}.log")
    log, buf, pid, t_end, next_s = None, "", None, time.time() + a.max_seconds, 0.0
    while not stop and time.time() < t_end:
        now = time.time()
        if log is None and os.path.exists(logfile):
            log = open(logfile, errors="replace")
        if log is not None:
            buf += log.read()
            *lines, buf = buf.split("\n")
            for line in lines:
                ev = parse_log_line(line)
                if ev:
                    out.write(json.dumps(dict({"kind": "ev", "t": round(now, 3)}, **ev)) + "\n")
        if now >= next_s:
            next_s = now + a.interval
            if pid is None or not os.path.exists(f"/proc/{pid}"):
                txt = read(pidfile).strip()
                pid = int(txt) if txt.isdigit() and os.path.exists(f"/proc/{txt}") else None
            texts = {"meminfo": read("/proc/meminfo"), "vmstat": read("/proc/vmstat"), "mountstats": read("/proc/self/mountstats"),
                     "netdev": read("/proc/net/dev"), "loadavg": read("/proc/loadavg"), "stat": read("/proc/stat")}
            for res in ("memory", "io", "cpu"):
                texts["psi." + res] = read("/proc/pressure/" + res)
            for dev in BLOCKS:
                texts["blk." + dev] = read(f"/sys/block/{dev}/stat")
            if pid:
                for n in ("stat", "io", "status"):
                    texts["pid." + n] = read(f"/proc/{pid}/{n}")
            c, g = flat_sample(texts)
            g["pid"] = pid
            out.write(json.dumps({"kind": "s", "t": round(now, 3), "took_ms": round((time.time() - now) * 1000, 1), "c": c, "g": g}) + "\n")
        time.sleep(0.25)
    out.write(json.dumps({"kind": "end", "t": time.time()}) + "\n")
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
