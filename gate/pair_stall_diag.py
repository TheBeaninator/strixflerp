#!/usr/bin/env python3
"""The diagnostic window of pod card PAIR-STALL c18e4b0bd2 (rank 1's fill_chunk frames went from 1.4 s to 5-20 s
under long warm appends on 2026-11-11 20:13-20:22Z and the pair died), run from control-node against workers F
(rank 0) and A (rank 1). It needs a window agreed with the pair's owner. `--plan` changes nothing: it reads the
serving conf, /health, the hold state and (over ssh, read-only) the workers' mounts, sizes and the live ranks' maps.

    python3 gate/pair_stall_diag.py --plan
    python3 gate/pair_stall_diag.py --minutes 25                 # arm S; arm L after it if S stalled and time is left
    python3 gate/pair_stall_diag.py --minutes 55 --arms S,L,E    # the named arms, as far as the time goes
    python3 gate/pair_stall_diag.py --analyze gate/results/pair-stall/<UTC>      # offline, from the collected files

Arms, one boot each of the deployed build (f64b5ff2e28b) from a COPY of the serving conf with lines appended (a later
assignment wins; serve_lib.sh sources the file), drafts on, greedy, PROMPT_CACHE_GIB=4 as served:
  S  as served but for RUN_ROOT, BENCH=0 and LEARN_DIR (a scratch directory on the node's /cache, never
     /cache/tfp-learned). The ladder of gate/prefill_dsv41_warm_ladder.py: the standard librarian prefix asked once
     (the 67,584-token state is learned into the scratch LEARN_DIR by the request itself: no state is copied from the
     served directory), its ladder twice, then a 4,096-token state of our own and its ladder until a rank 1
     fill_chunk frame exceeds 4 s or three passes are done. If the standard state is not resumed, a ~20k-token state
     of our own takes its place.
  L  the same with the ROCm libraries local: the files the live ranks map from WRAP0/WRAP1's LD_LIBRARY_PATH (and the
     .kpack directory beside lib/) copied to the scratch directory on the node's RAM overlay (worker /cache is
     mounted noexec, a shared object cannot be mapped from it), LD_LIBRARY_PATH = the copy, then the original.
  E  the same as S with ENGRAM_AIO=1 (TF_DS_ENGRAM_AIO on rank 0, --aio 1 on rank 1). The Engram shards (203 GB)
     do not fit a worker's /cache, so they are not staged.
While an arm runs, gate/pair_stall_sampler.py runs on both workers (a transient user unit, nice 5, /proc and /sys
only) and writes a sample every 2 s and every frame line of the rank's log with the time it appeared.

What it does on the pod: one pod-quad hold as agent over F and A; the launcher's serving run taken down;
scratch directories /home/pod/tfp/strixflerp-stall and /cache/strixflerp-stall on each worker; transient user units
of its own. On every exit path (the release time, an error, SIGTERM, SIGHUP, ^C): its units stopped, the samples
and rank logs collected into gate/results/pair-stall/<UTC>/, the scratch directories removed, THE HOLD RELEASED
FIRST, then up to 8 minutes of waiting for the pod's keeper to bring serving back (`strixfold-pair.sh up` is never
run from here: under a hold it fails). /health is recorded.

The analysis aligns rank 1's frames with the samples and judges leads L1-L5 by the thresholds in T below, which
were written before any run.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_nodes  # noqa: E402
import prefill_dsv41_pair as PP  # noqa: E402
import prefill_dsv41_warm_ladder as WL  # noqa: E402
from prefill_dsv41_pair import A, F, LAUNCHER, LIVE_CONF, ROOT, SSH, XDG, Run, now  # noqa: E402

S = "/home/pod/tfp/strixflerp-stall"          # on the node's RAM overlay: confs, runs, samples, the staged libraries
CS = "/cache/strixflerp-stall"                  # on the node's NVMe, as the served LEARN_DIR is: the learned states
HOME = "/home/pod"
DEPLOY = HOME + "/tfp/deploy/f64b5ff2e28b"
TOOLS = DEPLOY + "/tools/dsv41-hip"
PREFIX = "/data/pod/plans/strixfold/prefill-depth/librarian-dsv41-prefix.json"
OUT_ROOT = os.path.join(ROOT, "gate", "results", "pair-stall")
SAMPLER = os.path.join(ROOT, "gate", "pair_stall_sampler.py")
UNIT, SUNIT = "strixflerp-stall-up", "strixflerp-stall-sampler"
CARD = "c18e4b0bd2"
NEED_KEYS = ("WRAP0", "WRAP1", "ENGRAM_AIO", "ENGRAM_DIR", "LEARN_DIR", "PROMPT_CACHE_GIB", "RUN_ROOT", "DRAFTS", "BENCH",
             "MASTER_IP", "PEAK_GIB", "TEMPERATURE", "DEPLOY_DIR", "COLL_PROGRESS_MS")
MGMT_NET, RAIL_NET = "192.0.2.", "198.51.100."
PAIR = None         # the pair (pair_nodes), bound in main() with F and A from the live serving conf
BOOT_S = 200         # a boot: about 140 s of load, the probe, the checks
LEARN_S = 90         # the standard prefix read whole once (68,902 tokens: 71 s on 2026-11-11)
STD_PASS_S = 115     # one pass of the ladder on the standard state
SMALL_S = 170        # the 4,096-token state's seed and three clean passes
DOWN_S = 60          # an arm's stop and its files' collection
TAIL_S = 120         # stopping, collecting, removing the scratch directories and releasing, inside the window
MIN_ARM_S = BOOT_S + LEARN_S + STD_PASS_S + 75   # an arm is not started with less than this (+ DOWN_S + TAIL_S) left
# The libraries both live ranks mapped from the LD_LIBRARY_PATH directory on 2026-11-11 (used when no live rank can
# be read at the start of a window); relative to lib/.
DEFAULT_LIBS = """libamd_comgr.so.3.5.0 libamdhip64.so.7.17.26392-0000000 libhipblas.so.3.8 libhipblaslt.so.1.5
libhsa-amd-aqlprofile64.so.1.0.0 libhsa-runtime64.so.1.21.0 liborigami.so.1.0 librocblas.so.5.8 librocm_kpack.so.0.1.0
librocprofiler-register.so.0.6.0 librocroller.so.1.0.0 librocsolver.so.0.13 libroctx64.so.4.1.0
libtensilelite-host.so.1.0 llvm/lib/libLLVM.so.24.0git llvm/lib/libclang-cpp.so.24.0git
llvm/lib/x86_64-unknown-linux-gnu/libomp.so rocm_sysdeps/lib/librocm_sysdeps_bz2.so
rocm_sysdeps/lib/librocm_sysdeps_drm.so.2 rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1
rocm_sysdeps/lib/librocm_sysdeps_elf.so.1 rocm_sysdeps/lib/librocm_sysdeps_liblzma.so.5.8.1
rocm_sysdeps/lib/librocm_sysdeps_numa.so.1 rocm_sysdeps/lib/librocm_sysdeps_z.so.1.3.2
rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7""".split()

# ---- the decision thresholds, written on 2026-11-11 before any run of this script ---------------------------------
# A frame is rank 1's fill_chunk frame (its log line), from (the line's time - its ms) to the line's time. Rates are
# per second of frame time of a class; "busy" is the READ operations' summed execute time per second of frame time
# (0.30 = reads in flight 30% of the time; above 1 with several at once). All on rank 1's node unless said.
T = {
    "stall_ms": 4000,              # stalled: a frame longer than this (the card's criterion is every frame under 3 s)
    "normal_ms": 3000,             # normal: a frame of at most this; between the two: neither class
    "min_stalled_frames": 3,       # with fewer stalled frames L1-L3 are "not decided"
    # L1: file pages of the libraries (management NFS mount) or of the netboot root image (loop0) re-read under pressure
    "l1_busy": 0.30,               # supported: library-mount or root-image read busy at least this in stalled frames,
    "l1_majflt_s": 5.0,            #   and the rank's major faults at least this a second
    "l1_refault_s": 250.0,         #   or file refaults (workingset_refault_file) at least this many pages a second,
    "l1_ratio": 3.0,               #   and that busy at least this many times the normal frames'
    "l1_out_busy": 0.05,           # ruled out: both busies under this and both under l1_out_mb_s in stalled frames
    "l1_out_mb_s": 1.0,
    # L2: Engram rows read on demand over the rail NFS mount inside the prefill step
    "l2_busy": 0.30,               # supported: Engram-mount read busy at least this in stalled frames, and
    "l2_ratio": 3.0,               #   that busy at least this many times the normal frames' (reads that take the
                                   #   share of a frame they always took do not explain a longer frame)
    "l2_out_busy": 0.05,           # ruled out: Engram-mount read busy under this in stalled frames
    # L3: memory pressure or compaction with no NFS reads in flight
    "l3_psi_some": 0.10,           # supported: PSI memory "some" at least this share of stalled frame time, or
    "l3_direct_s": 1000.0,         #   direct reclaim scanning at least this many pages a second, or
    "l3_compact_s": 1.0,           #   compaction stalls at least this many a second; and neither L1 nor L2 busy
    "l3_out_psi": 0.01,            # ruled out: PSI memory some under this, direct scan under l3_out_direct_s and
    "l3_out_direct_s": 10.0,       #   compaction stalls under l3_out_compact_s
    "l3_out_compact_s": 0.1,
    # L4: the prompt cache evicting (rank 0's "prompt cache:" lines)
    #   supported: the first eviction is logged before the first stalled frame; ruled out: stalled frames with evicted 0
    # L5: collectives on the management NIC. Over all frames: NIC bytes (rx + tx) less the NFS mounts' read bytes
    # through that NIC = the ranks' own traffic.
    "l5_share": 0.20,              # supported: the management NIC carries at least this share of it
    "l5_out_share": 0.05,          # ruled out: under this share
    "l5_min_mb_s": 1.0,            # not decided when the ranks' traffic is under this (nothing to apportion)
    # the arms against each other, same ladder: L (libraries local) or E (Engram AIO) "removes" the stall when S had at
    # least min_stalled_frames stalled frames and the arm, over at least as many frames, had none
}


# ---------------------------------------------------------------- pure helpers (tests/test_pair_stall.py)
def conf_values(text):
    """KEY=value lines of a serve conf -> {key: value}: the last assignment wins, quotes and trailing comments off."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"([A-Z][A-Z0-9_]*)=(.*)$", line)
        if not m:
            continue
        v = m.group(2)
        if v[:1] in "'\"" and v[:1] and v.find(v[0], 1) > 0:
            v = v[1:v.find(v[0], 1)]
        else:
            v = v.split("#", 1)[0].strip()
        out[m.group(1)] = v
    return out


def lib_dir_of(wrap):
    """The first LD_LIBRARY_PATH entry of a WRAP value, or None."""
    m = re.search(r"LD_LIBRARY_PATH=([^\s:]+)", wrap or "")
    return m.group(1) if m else None


def overrides(arm, live):
    """The lines an arm appends to the copy of the serving conf (`live`: its values)."""
    over = {"BENCH": "0", "DRAFTS": "1", "PROMPT_CACHE_GIB": live.get("PROMPT_CACHE_GIB", "4"),
            "RUN_ROOT": "$HOME/tfp/strixflerp-stall/run", "LEARN_DIR": f"{CS}/learned-{arm}"}
    if arm == "L":
        lib = lib_dir_of(live.get("WRAP1", ""))
        for k in ("WRAP0", "WRAP1"):
            over[k] = live[k].replace("LD_LIBRARY_PATH=" + lib, f"LD_LIBRARY_PATH={S}/therock/lib:{lib}")
    if arm == "E":
        over["ENGRAM_AIO"] = "1"
    return over


def conf_text(text, arm):
    over = overrides(arm, conf_values(text))
    return text + f"\n# --- strixflerp PAIR-STALL window, arm {arm}: appended, a later assignment wins ---\n" + \
        "".join(f"{k}='{v}'\n" for k, v in over.items())


def mount_of(path, mounts):
    """The mount a path is on: the longest mount point that is a prefix. mounts: [(device, point, fstype, options)]."""
    best = None
    for m in mounts:
        p = m[1].rstrip("/") or "/"
        if (path == p or path.startswith(p.rstrip("/") + "/") or p == "/") and (best is None or len(p) > len(best[1].rstrip("/") or "/")):
            best = m
    return best


def target_of(findmnt_row):
    """The mount point of a "SOURCE TARGET FSTYPE" row, or None."""
    parts = (findmnt_row or "").split()
    return parts[1] if len(parts) >= 2 else None


def origins(paths, lib_mount="/data/models", rail_mount="/data/models-rail"):
    """Where a rank's mapped or open files are: counts by origin."""
    out = {}
    for p in paths:
        if p.startswith(rail_mount + "/"):
            k = "rail-nfs"
        elif p.startswith(lib_mount + "/"):
            k = "mgmt-nfs"
        elif p.startswith(S + "/"):
            k = "scratch"
        elif p.startswith("/cache/"):
            k = "cache"
        else:
            k = "root-or-ram"
        out[k] = out.get(k, 0) + 1
    return out


def load_samples(path):
    """A sampler file -> (header, samples sorted by time, events)."""
    hdr, samples, events = {}, [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue                                    # a line cut by the stop
            if d.get("kind") == "hdr" and not hdr:
                hdr = d
            elif d.get("kind") == "s":
                samples.append(d)
            elif d.get("kind") == "ev":
                events.append(d)
    samples.sort(key=lambda d: d["t"])
    return hdr, samples, events


def delta(samples, key, a, b):
    """A cumulative counter's increase over [a, b], each sample interval's increase shared by its overlap with [a, b]
    (a counter that fell, a new process, counts as 0 for that interval). None when no sample interval overlaps."""
    total, seen = 0.0, False
    for s0, s1 in zip(samples, samples[1:]):
        t0, t1 = s0["t"], s1["t"]
        ov = min(b, t1) - max(a, t0)
        if ov <= 0 or t1 <= t0:
            continue
        v0, v1 = s0["c"].get(key), s1["c"].get(key)
        if v0 is None or v1 is None:
            continue
        seen = True
        total += max(0, v1 - v0) * ov / (t1 - t0)
    return total if seen else None


def gauge_at(samples, key, t):
    """The gauge's value in the last sample at or before t (the first sample's before any)."""
    val = None
    for s in samples:
        if s["t"] > t and val is not None:
            break
        if key in s["g"]:
            val = s["g"][key]
    return val


def frames_of(events, kind="fill_chunk"):
    """Rank 1's frames: [{"a": start, "b": end, "ms", "step", "cls": "stalled" | "normal" | "slow"}]."""
    out = []
    for e in events:
        if e.get("ev") == "frame" and e.get("frame") == kind:
            cls = "stalled" if e["ms"] > T["stall_ms"] else "normal" if e["ms"] <= T["normal_ms"] else "slow"
            out.append({"a": e["t"] - e["ms"] / 1000.0, "b": e["t"], "ms": e["ms"], "step": e.get("step"), "cls": cls})
    return out


def roles(hdr):
    """The node's interfaces and NFS mounts by network: management (192.0.2.x) and rail (198.51.100.x)."""
    r = {"mgmt_if": None, "rail_if": None, "mgmt_mounts": [], "rail_mounts": []}
    for ifc, ip in (hdr.get("ifaces") or {}).items():
        if ip and ip.startswith(MGMT_NET):
            r["mgmt_if"] = ifc
        elif ip and ip.startswith(RAIL_NET):
            r["rail_if"] = ifc
    for mnt, dev in (hdr.get("mounts") or {}).items():
        if dev.startswith(MGMT_NET):
            r["mgmt_mounts"].append(mnt)
        elif dev.startswith(RAIL_NET):
            r["rail_mounts"].append(mnt)
    return r


def class_metrics(samples, hdr, frames, lib_mount="/data/models", eng_mount="/data/models-rail", shift=0.0):
    """The rates of one class of frames from one node's samples. shift: seconds added to the frames' times (the
    other node's clock). None when the frames have no samples."""
    if not frames:
        return None
    r = roles(hdr)
    secs = sum(f["b"] - f["a"] for f in frames)
    if secs <= 0:
        return None

    def tot(key):
        vals = [delta(samples, key, f["a"] + shift, f["b"] + shift) for f in frames]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    def rate(key, scale=1.0):
        v = tot(key)
        return None if v is None else v * scale / secs

    def per(num, den):
        n, d = tot(num), tot(den)
        return None if n is None or not d else n / d

    def summed(keys, scale):
        vals = [tot(k) for k in keys]
        vals = [v for v in vals if v is not None]
        return sum(vals) * scale / secs if vals else None

    def mean_gauge(key, scale=1.0):
        vals = [gauge_at(samples, key, f["b"] + shift) for f in frames]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) * scale / len(vals) if vals else None

    m = {"frames": len(frames), "seconds": round(secs, 2), "frame_ms_mean": round(sum(f["ms"] for f in frames) / len(frames))}
    for tag, mnt in (("lib", lib_mount), ("eng", eng_mount)):
        m[tag + "_mb_s"] = rate(f"nfs:{mnt}:server_read_bytes", 1e-6)
        m[tag + "_busy"] = rate(f"nfs:{mnt}:read_exe_ms", 1e-3)
        m[tag + "_reads_s"] = rate(f"nfs:{mnt}:read_ops")
        m[tag + "_rtt_ms_op"] = per(f"nfs:{mnt}:read_rtt_ms", f"nfs:{mnt}:read_ops")
        m[tag + "_exe_ms_op"] = per(f"nfs:{mnt}:read_exe_ms", f"nfs:{mnt}:read_ops")
    b = tot(f"nfs:{eng_mount}:server_read_bytes")
    m["eng_mb_frame"] = None if b is None else b * 1e-6 / len(frames)
    m["root_mb_s"] = rate("blk:loop0:read_sectors", 512e-6)
    m["root_busy"] = rate("blk:loop0:read_ms", 1e-3)
    m["pid_majflt_s"] = rate("pid:majflt")
    m["vm_majflt_s"] = rate("vm:pgmajfault")
    m["refault_file_s"] = rate("vm:workingset_refault_file")
    m["pgscan_kswapd_s"] = rate("vm:pgscan_kswapd")
    m["pgscan_direct_s"] = rate("vm:pgscan_direct")
    m["compact_stall_s"] = rate("vm:compact_stall")
    for res in ("memory", "io", "cpu"):
        for k in ("some", "full"):
            m[f"psi_{res}_{k}"] = rate(f"psi:{res}:{k}_us", 1e-6)
    m["pid_read_mb_s"] = rate("pid:read_bytes", 1e-6)
    m["pid_rchar_mb_s"] = rate("pid:rchar", 1e-6)
    for tag in ("mgmt", "rail"):
        ifc = r[tag + "_if"]
        m[tag + "_nic_mb_s"] = summed([f"net:{ifc}:rx_bytes", f"net:{ifc}:tx_bytes"], 1e-6) if ifc else None
        m[tag + "_nfs_mb_s"] = summed([f"nfs:{mnt}:server_read_bytes" for mnt in r[tag + "_mounts"]], 1e-6)
    m["mem_available_gib"] = mean_gauge("mem:MemAvailable", 1 / 1048576)
    m["load1"] = mean_gauge("load1")
    m["procs_blocked"] = mean_gauge("procs_blocked")
    m["pid_rss_file_mb"] = mean_gauge("pid:RssFile", 1 / 1024)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def _v(m, k):
    return (m or {}).get(k) or 0.0


def judge(stalled, normal, allf, cache_events=(), frames=(), cache_shift=0.0):
    """The leads from rank 1's node's metrics of the stalled, the normal and all frames, and rank 0's "prompt
    cache:" events. {lead: [verdict, reason]}; verdict: "supported", "ruled out" or "not decided"."""
    out = {}
    enough = stalled is not None and stalled["frames"] >= T["min_stalled_frames"]
    few = f"fewer than {T['min_stalled_frames']} stalled frames"
    # L1
    if not enough:
        out["L1"] = ["not decided", few]
    else:
        busy = max(_v(stalled, "lib_busy"), _v(stalled, "root_busy"))
        nbusy = max(_v(normal, "lib_busy"), _v(normal, "root_busy"))
        faults = _v(stalled, "pid_majflt_s") >= T["l1_majflt_s"] or _v(stalled, "refault_file_s") >= T["l1_refault_s"]
        why = (f"stalled frames: library mount busy {_v(stalled, 'lib_busy'):.2f} at {_v(stalled, 'lib_mb_s'):.1f} MB/s, root image busy "
               f"{_v(stalled, 'root_busy'):.2f} at {_v(stalled, 'root_mb_s'):.1f} MB/s, rank major faults {_v(stalled, 'pid_majflt_s'):.1f}/s, "
               f"file refaults {_v(stalled, 'refault_file_s'):.0f}/s; normal frames' busy {nbusy:.2f}")
        if busy >= T["l1_busy"] and faults and busy >= T["l1_ratio"] * nbusy:
            out["L1"] = ["supported", why]
        elif busy < T["l1_out_busy"] and max(_v(stalled, "lib_mb_s"), _v(stalled, "root_mb_s")) < T["l1_out_mb_s"]:
            out["L1"] = ["ruled out", why]
        else:
            out["L1"] = ["not decided", why]
    # L2
    if not enough:
        out["L2"] = ["not decided", few]
    else:
        busy = _v(stalled, "eng_busy")
        why = (f"stalled frames: Engram mount busy {busy:.2f} (normal {_v(normal, 'eng_busy'):.2f}), {_v(stalled, 'eng_mb_frame'):.1f} MB a frame, "
               f"{_v(stalled, 'eng_exe_ms_op'):.1f} ms a READ (rtt {_v(stalled, 'eng_rtt_ms_op'):.1f}); normal: "
               f"{_v(normal, 'eng_mb_frame'):.1f} MB a frame, {_v(normal, 'eng_exe_ms_op'):.1f} ms a READ (rtt {_v(normal, 'eng_rtt_ms_op'):.1f})")
        if busy >= T["l2_busy"] and busy >= T["l2_ratio"] * _v(normal, "eng_busy"):
            out["L2"] = ["supported", why]
        elif busy < T["l2_out_busy"]:
            out["L2"] = ["ruled out", why]
        else:
            out["L2"] = ["not decided", why]
    # L3
    if not enough:
        out["L3"] = ["not decided", few]
    else:
        psi, direct, comp = _v(stalled, "psi_memory_some"), _v(stalled, "pgscan_direct_s"), _v(stalled, "compact_stall_s")
        nfs_busy = max(_v(stalled, "lib_busy"), _v(stalled, "root_busy"), _v(stalled, "eng_busy"))
        why = (f"stalled frames: PSI memory some {psi:.3f} full {_v(stalled, 'psi_memory_full'):.3f}, PSI io some "
               f"{_v(stalled, 'psi_io_some'):.3f}, direct scan {direct:.0f} pages/s, kswapd scan {_v(stalled, 'pgscan_kswapd_s'):.0f} pages/s, "
               f"compaction stalls {comp:.2f}/s, NFS/root read busy {nfs_busy:.2f}")
        pressure = psi >= T["l3_psi_some"] or direct >= T["l3_direct_s"] or comp >= T["l3_compact_s"]
        if pressure and nfs_busy < min(T["l1_busy"], T["l2_busy"]):
            out["L3"] = ["supported", why]
        elif pressure:
            out["L3"] = ["not decided", why + ": pressure with NFS reads in flight is L1 or L2's mechanism, not independent of NFS"]
        elif psi < T["l3_out_psi"] and direct < T["l3_out_direct_s"] and comp < T["l3_out_compact_s"]:
            out["L3"] = ["ruled out", why]
        else:
            out["L3"] = ["not decided", why]
    # L4
    first_stall = min((f["a"] for f in frames if f["cls"] == "stalled"), default=None)
    evs = sorted((e for e in cache_events if e.get("ev") == "cache"), key=lambda e: e["t"])
    first_evict = next((e["t"] - cache_shift for e in evs if e["evicted"] > 0), None)
    last = evs[-1] if evs else None
    state = f"last line: {last['states']} states, {last['mib']} of {last['cap_mib']} MiB, evicted {last['evicted']}" if last else "no prompt cache lines"
    if first_stall is None:
        out["L4"] = ["not decided", "no stalled frame; " + state]
    elif not evs:
        out["L4"] = ["not decided", state]
    elif first_evict is None or first_evict > first_stall:
        out["L4"] = ["ruled out", "frames stalled before any eviction; " + state]
    else:
        out["L4"] = ["supported", f"the first eviction was logged {first_stall - first_evict:.0f} s before the first stalled frame; " + state]
    # L5
    if allf is None:
        out["L5"] = ["not decided", "no frames"]
    else:
        mg = max(0.0, _v(allf, "mgmt_nic_mb_s") - _v(allf, "mgmt_nfs_mb_s"))
        rl = max(0.0, _v(allf, "rail_nic_mb_s") - _v(allf, "rail_nfs_mb_s"))
        why = (f"all frames: management NIC {_v(allf, 'mgmt_nic_mb_s'):.1f} MB/s ({_v(allf, 'mgmt_nfs_mb_s'):.1f} of it NFS reads), "
               f"rail NIC {_v(allf, 'rail_nic_mb_s'):.1f} MB/s ({_v(allf, 'rail_nfs_mb_s'):.1f} NFS)")
        if mg + rl < T["l5_min_mb_s"] or allf.get("rail_nic_mb_s") is None:
            out["L5"] = ["not decided", why]
        elif mg / (mg + rl) >= T["l5_share"]:
            out["L5"] = ["supported", why]
        elif mg / (mg + rl) < T["l5_out_share"]:
            out["L5"] = ["ruled out", why]
        else:
            out["L5"] = ["not decided", why]
    return out


def contrast(arms):
    """The arms against each other: {arm: sentence} for L and E when S is there. arms: {tag: frames list}."""
    out = {}
    s = arms.get("S")
    if s is None:
        return out
    ns = sum(1 for f in s if f["cls"] == "stalled")
    for tag, lead in (("L", "L1"), ("E", "L2")):
        if tag not in arms:
            continue
        n = sum(1 for f in arms[tag] if f["cls"] == "stalled")
        if ns < T["min_stalled_frames"]:
            out[tag] = f"arm S had {ns} stalled frame(s): arm {tag} ({n} stalled of {len(arms[tag])}) says nothing about {lead}"
        elif n == 0 and len(arms[tag]) >= len(s):
            out[tag] = f"arm {tag} removed the stall ({ns} stalled of {len(s)} frames in S, 0 of {len(arms[tag])} in {tag}): supports {lead}"
        elif n == 0:
            out[tag] = f"arm {tag} had no stalled frame but ran fewer frames ({len(arms[tag])} against {len(s)}): {lead} not decided by the arms"
        else:
            out[tag] = f"arm {tag} stalled too ({n} of {len(arms[tag])} frames; S {ns} of {len(s)}): the stall does not need what {tag} removed, against {lead} as the only cause"
    return out


ROWS = (("frames", "frames"), ("seconds", "frame seconds"), ("frame_ms_mean", "mean frame ms"),
        ("lib_mb_s", "library mount (mgmt NFS) read MB/s"), ("lib_busy", "library mount READ busy"),
        ("lib_rtt_ms_op", "library mount ms rtt a READ"), ("lib_exe_ms_op", "library mount ms execute a READ"),
        ("eng_mb_s", "Engram mount (rail NFS) read MB/s"), ("eng_mb_frame", "Engram mount MB a frame"), ("eng_busy", "Engram mount READ busy"),
        ("eng_rtt_ms_op", "Engram mount ms rtt a READ"), ("eng_exe_ms_op", "Engram mount ms execute a READ"),
        ("root_mb_s", "root image (loop0) read MB/s"), ("root_busy", "root image read busy"),
        ("pid_majflt_s", "rank major faults /s"), ("vm_majflt_s", "node major faults /s"), ("refault_file_s", "file refaults pages/s"),
        ("psi_memory_some", "PSI memory some"), ("psi_memory_full", "PSI memory full"), ("psi_io_some", "PSI io some"), ("psi_io_full", "PSI io full"),
        ("psi_cpu_some", "PSI cpu some"), ("pgscan_kswapd_s", "kswapd scan pages/s"), ("pgscan_direct_s", "direct scan pages/s"),
        ("compact_stall_s", "compaction stalls /s"), ("mgmt_nic_mb_s", "management NIC MB/s"), ("mgmt_nfs_mb_s", "  of it NFS reads MB/s"),
        ("rail_nic_mb_s", "rail NIC MB/s"), ("rail_nfs_mb_s", "  of it NFS reads MB/s"), ("pid_read_mb_s", "rank read_bytes MB/s"),
        ("pid_rchar_mb_s", "rank rchar MB/s"), ("mem_available_gib", "MemAvailable GiB"), ("pid_rss_file_mb", "rank RssFile MB"),
        ("load1", "load average"), ("procs_blocked", "blocked processes"))


def analyze(d):
    """The collected directory -> {"arms": {tag: {...}}, "contrast": {...}}."""
    doc = {}
    if os.path.isfile(os.path.join(d, "result.json")):
        with open(os.path.join(d, "result.json")) as f:
            doc = json.load(f)
    lib_mount, eng_mount = doc.get("lib_mount") or "/data/models", doc.get("engram_mount") or "/data/models-rail"
    tags = sorted({n.split("-rank")[0] for n in os.listdir(d) if n.endswith("-rank1.samples.jsonl")})
    out, allframes = {"dir": d, "thresholds": T, "arms": {}}, {}
    for tag in tags:
        h1, s1, e1 = load_samples(os.path.join(d, f"{tag}-rank1.samples.jsonl"))
        p0 = os.path.join(d, f"{tag}-rank0.samples.jsonl")
        h0, s0, e0 = load_samples(p0) if os.path.isfile(p0) else ({}, [], [])
        row = next((a for a in doc.get("arms", []) if a.get("arm") == tag), {})
        off = row.get("clock_offset_s") or {}
        shift = (off.get("rank0") or 0.0) - (off.get("rank1") or 0.0)       # rank 1's clock -> rank 0's
        frames = frames_of(e1)
        allframes[tag] = frames
        by = {c: [f for f in frames if f["cls"] == c] for c in ("stalled", "normal")}
        node1 = {c: class_metrics(s1, h1, by[c], lib_mount, eng_mount) for c in by}
        node1["all"] = class_metrics(s1, h1, frames, lib_mount, eng_mount)
        node0 = {c: class_metrics(s0, h0, by[c], lib_mount, eng_mount, shift) for c in by} if s0 else {}
        gaps = [b["t"] - a["t"] for a, b in zip(s1, s1[1:])]
        out["arms"][tag] = {
            "frames": len(frames), "stalled": len(by["stalled"]), "normal": len(by["normal"]),
            "frame_ms": [f["ms"] for f in frames], "rank1_node": node1, "rank0_node": node0,
            "sampler": {"rank1_samples": len(s1), "rank0_samples": len(s0), "mlocked": [h1.get("mlocked"), h0.get("mlocked")],
                        "rank1_longest_gap_s": round(max(gaps), 2) if gaps else None},
            "leads": judge(node1["stalled"], node1["normal"], node1["all"], e0, frames, shift),
        }
    out["contrast"] = contrast(allframes)
    return out


def render(an):
    lines = [f"PAIR-STALL {CARD} analysis of {an['dir']}",
             f"a frame is stalled over {T['stall_ms']} ms and normal up to {T['normal_ms']} ms; rates are per second of frame time"]

    def fmt(v):
        return "-" if v is None else f"{v:,.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:,.1f}" if isinstance(v, float) else f"{v:,}"

    for tag, a in an["arms"].items():
        lines.append(f"\n== arm {tag}: {a['frames']} fill_chunk frames, {a['stalled']} stalled, {a['normal']} normal; sampler "
                     f"{a['sampler']['rank1_samples']} samples on rank 1's node (longest gap {a['sampler']['rank1_longest_gap_s']} s, "
                     f"mlocked {a['sampler']['mlocked']})")
        n1, n0 = a["rank1_node"], a["rank0_node"]
        lines.append(f"{'':42s} {'rank 1 node':>25s} {'rank 0 node':>25s}")
        lines.append(f"{'':42s} {'normal':>12s} {'stalled':>12s} {'normal':>12s} {'stalled':>12s}")
        for key, label in ROWS:
            cells = [fmt((n.get(c) or {}).get(key)) for n in (n1, n0) for c in ("normal", "stalled")]
            lines.append(f"{label:42s} " + " ".join(f"{c:>12s}" for c in cells))
        for lead, (verdict, why) in a["leads"].items():
            lines.append(f"{lead}: {verdict.upper()}: {why}")
    for tag, text in an.get("contrast", {}).items():
        lines.append(f"arms: {text}")
    if not an["arms"]:
        lines.append("no <arm>-rank1.samples.jsonl files here")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the window
class Stop(Exception):
    pass


class Window(Run):
    def say(self, text):                     # the terminal may be gone (SIGHUP): the teardown must still run
        try:
            super().say(text)
        except OSError:
            pass


def probe_script(lib, engram, peer_rail=None, rails=()):
    """A read-only look at a worker (bash, over ssh): key=value lines. rails: the pair's rail addresses (routes
    read); peer_rail: the other rank's, whose route names this node's rail NIC (rail_nic=)."""
    targets = " ".join([MGMT_NET + "31", RAIL_NET + "31"] + [x for x in rails if x])
    nic = f"echo rail_nic=$({pair_nodes.rail_nic_cmd(peer_rail)})" if peer_rail else ""
    return f"""
LIB={shlex.quote(lib)}; ENG={shlex.quote(engram)}
echo host=$(hostname)
echo lib_real=$(readlink -f $LIB)
echo lib_mount=$(findmnt -n -o SOURCE,TARGET,FSTYPE -T $LIB | tr -s ' ')
echo lib_bytes=$(du -sb $LIB 2>/dev/null | cut -f1)
echo kpack_bytes=$(du -sb $LIB/../.kpack 2>/dev/null | cut -f1)
echo eng_real=$(readlink -f $ENG)
echo eng_mount=$(findmnt -n -o SOURCE,TARGET,FSTYPE -T $(readlink -f $ENG) | tr -s ' ')
echo eng_bytes=$(du -sbL $ENG 2>/dev/null | cut -f1)
echo cache_mount=$(findmnt -n -o SOURCE,FSTYPE,OPTIONS -T /cache | tr -s ' ')
echo cache_avail_kb=$(df -k --output=avail /cache | tail -1 | tr -d ' ')
echo cache_writable=$(test -w /cache && echo 1 || echo 0)
echo ram_mount=$(findmnt -n -o SOURCE,FSTYPE,OPTIONS -T {HOME}/tfp | tr -s ' ' | cut -c1-80)
echo ram_avail_kb=$(df -k --output=avail {HOME}/tfp | tail -1 | tr -d ' ')
echo mem_available_kb=$(awk '/^MemAvailable:/ {{print $2}}' /proc/meminfo)
echo swap_total_kb=$(awk '/^SwapTotal:/ {{print $2}}' /proc/meminfo)
echo root_image=$(cat /sys/block/loop0/loop/backing_file 2>/dev/null) from $(tr ' ' '\\n' < /proc/cmdline | grep ^nfsroot=)
echo scratch_exists=$(test -e {S} -o -e {CS} && echo 1 || echo 0)
echo own_units=$(XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user list-units 'strixflerp-stall*' --no-legend --plain | wc -l)
echo first_light=$(test -x {TOOLS}/first_light.sh && echo 1 || echo 0)
echo python=$(python3 -V 2>&1)
echo memlock_default=$(XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user show -p DefaultLimitMEMLOCK --value)
ip -o -4 addr show | awk '$2 != "lo" {{print "addr=" $2 " " $4}}'
for t in {targets}; do echo "route=$t $(ip -o route get $t | tr -s ' ' | cut -c1-90)"; done
{nic}
p=$(pgrep -x tf-dsv41-lanes | head -1); [ -z "$p" ] && p=$(pgrep -x tensorfold-nati | head -1)
echo rank_pid=$p
if [ -n "$p" ]; then
  awk '$6 ~ /^\\// {{print $6}}' /proc/$p/maps | sort -u | while read f; do echo "map=$(stat -L -c %s "$f" 2>/dev/null) $f"; done
  ls -l /proc/$p/fd 2>/dev/null | awk '{{print $NF}}' | grep '^/' | grep -v '^/dev\\|^/proc\\|^/sys\\|^/run' | sort -u | sed 's/^/fd=/'
  ss -tnp 2>/dev/null | grep "pid=$p," | awk '{{print "sock=" $4 " " $5}}'
  tr '\\0' '\\n' < /proc/$p/environ | grep -E '^(LD_LIBRARY_PATH|TF_DS_ENGRAM_AIO|TF_TP_MASTER)=' | sed 's/^/env=/'
  echo "aio_arg=$(tr '\\0' ' ' < /proc/$p/cmdline | grep -o -- '--aio [01]')"
  grep -E '^Rss(File|Anon|Shmem)' /proc/$p/status | tr -s '\\t ' ' ' | sed 's/^/rss=/'
fi
grep -n 'ENGRAM_AIO\\|wv=WRAP\\|wrap\\[@\\]}}" "${{SERVE_CMD\\|--learn-dir\\|--prompt-cache-gib\\|TF_DS_ENGRAM=' {TOOLS}/serve_lib.sh | cut -c1-150 | sed 's/^/cite=serve_lib.sh:/'
"""


def probe(ip, lib, engram, peer_rail=None, rails=()):
    """{key: value or [values]} from a worker, or {"error": ...}. Read-only."""
    try:
        p = subprocess.run(SSH + [f"pod@{ip}", "bash -s"], input=probe_script(lib, engram, peer_rail, rails), capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if p.returncode != 0 and not p.stdout:
        return {"error": (p.stderr or "ssh failed").strip()[-200:]}
    out = {}
    for line in p.stdout.splitlines():
        k, sep, v = line.partition("=")
        if not sep:
            continue
        if k in ("map", "fd", "sock", "addr", "route", "cite", "env", "rss"):
            out.setdefault(k, []).append(v)
        else:
            out[k] = v
    return out


def lib_list(pr, lib_real):
    """The files (relative to lib/) a probed rank maps from the library directory, with their bytes; the default
    list when no rank was read."""
    rel = {}
    for row in pr.get("map", []):
        size, _, path = row.partition(" ")
        if path.startswith(lib_real.rstrip("/") + "/"):
            rel[path[len(lib_real.rstrip("/")) + 1:]] = int(size or 0)
    return rel


def plan(arms, minutes, quiet=False, busy_ok=False):
    """Everything the window would do, and its preconditions, read-only. (ok, facts)."""
    bad, facts = [], {}

    def p(text=""):
        if not quiet:
            print(text)

    def fail(text):
        bad.append(text)
        p("FAIL: " + text)

    p(f"PAIR-STALL {CARD} diagnostic window: the plan ({now()}); nothing is changed by --plan")
    p(pair_nodes.describe(PAIR))
    facts["pair"] = PAIR
    try:
        with open(LIVE_CONF) as f:
            text = f.read()
    except OSError as e:
        fail(f"serving conf {LIVE_CONF}: {e}")
        return False, facts
    live = conf_values(text)
    missing = [k for k in NEED_KEYS if k not in live]
    if missing:
        fail(f"the serving conf has no {', '.join(missing)}")
        return False, facts
    lib, engram = lib_dir_of(live["WRAP1"]), live["ENGRAM_DIR"].replace("$HOME", HOME)
    if not lib or lib_dir_of(live["WRAP0"]) != lib:
        fail(f"WRAP0/WRAP1 do not name one LD_LIBRARY_PATH: {live['WRAP0']!r} {live['WRAP1']!r}")
        return False, facts
    p(f"serving conf {LIVE_CONF}: keys present ({', '.join(NEED_KEYS)})")
    p(f"  as served: PROMPT_CACHE_GIB={live['PROMPT_CACHE_GIB']} DRAFTS={live['DRAFTS']} TEMPERATURE={live['TEMPERATURE']} ENGRAM_AIO={live['ENGRAM_AIO']} "
      f"LEARN_DIR={live['LEARN_DIR']} MASTER_IP={live['MASTER_IP']} COLL_PROGRESS_MS={live['COLL_PROGRESS_MS']} PEAK_GIB={live['PEAK_GIB']}")
    p(f"  WRAP1={live['WRAP1']}")
    if live["PROMPT_CACHE_GIB"] != "4" or live["DRAFTS"] != "1" or live["TEMPERATURE"] != "0":
        fail("the serving conf is not the one of the stall (cache 4 GiB, drafts on, greedy)")
    if DEPLOY.rsplit("/", 1)[1] not in live["DEPLOY_DIR"]:
        fail(f"the serving conf's DEPLOY_DIR is {live['DEPLOY_DIR']}, this script boots {DEPLOY}")
    for name in (PREFIX, SAMPLER):
        if not os.path.isfile(name):
            fail(f"missing {name}")
    for ip in (F, A):
        rc = subprocess.run(["pod-quad", "held", ip], capture_output=True, text=True, timeout=30).returncode
        if rc == 0:
            fail(f"{ip} is under a pod-quad hold")
    h = Run("/tmp", 0).health()
    livec = (h or {}).get("live", {})
    p(f"/health: {json.dumps(h)[:300]}")
    if not h or h.get("warming") is not False:
        fail("the pair does not answer /health as ready")
    elif livec.get("connections", 1) != 0 or livec.get("waiting", 1) != 0:
        if busy_ok:
            p(f"the pair is not idle ({livec}): --busy-ok")
        else:
            fail(f"the pair is not idle: {livec} (--busy-ok to go on anyway)")
    staged = None
    rails = (PAIR["rank0"]["rail"], PAIR["rank1"]["rail"])
    for ip, rank in ((F, 0), (A, 1)):
        pr = probe(ip, lib, engram, rails[1 - rank], rails)
        PAIR[f"rank{rank}"]["rail_nic"] = pr.get("rail_nic") or None
        facts[f"rank{rank}"] = pr
        if "error" in pr or "lib_real" not in pr:
            fail(f"{ip}: no probe ({pr.get('error', 'incomplete')})")
            continue
        p(f"\nrank {rank} on {pr.get('host')} ({ip}), live rank pid {pr.get('rank_pid') or 'none'}:")
        libs = lib_list(pr, pr["lib_real"]) or None
        nbytes = sum(libs.values()) if libs else None
        staged = max(staged or 0, (nbytes or 600 << 20) + int(pr.get("kpack_bytes") or 0))
        p(f"  libraries: {lib} is on [{pr.get('lib_mount')}]: whole directory {int(pr.get('lib_bytes') or 0) / 2**30:.2f} GiB; the live rank maps "
          f"{len(libs) if libs else '?'} files of it, {(nbytes or 0) / 2**20:.0f} MiB; .kpack {int(pr.get('kpack_bytes') or 0) / 2**20:.0f} MiB")
        on_mgmt = (pr.get("lib_mount") or "").startswith(MGMT_NET)
        p(f"  -> the libraries are {'ON' if on_mgmt else 'NOT on'} the management network's NFS server (L1's premise); "
          f"route to it: {next((r for r in pr.get('route', []) if r.startswith(MGMT_NET + '31')), '?')}")
        p(f"  Engram: {engram} -> {pr.get('eng_real')} on [{pr.get('eng_mount')}], {int(pr.get('eng_bytes') or 0) / 2**30:.1f} GiB; "
          f"live: {' '.join(pr.get('env', []))} {pr.get('aio_arg', '')}")
        p(f"  /cache: [{pr.get('cache_mount')}] {int(pr.get('cache_avail_kb') or 0) / 2**20:.1f} GiB free, writable {pr.get('cache_writable')}")
        p(f"  RAM overlay under {HOME}/tfp: [{pr.get('ram_mount')}] {int(pr.get('ram_avail_kb') or 0) / 2**20:.1f} GiB free; MemAvailable "
          f"{int(pr.get('mem_available_kb') or 0) / 2**20:.1f} GiB with the model loaded; swap {pr.get('swap_total_kb')} kB; root image {pr.get('root_image')}")
        p(f"  interfaces: {'; '.join(pr.get('addr', []))}; rail NIC (route to the peer's rail {rails[1 - rank]}): {pr.get('rail_nic') or '?'}")
        p(f"  the live rank's sockets: {'; '.join(pr.get('sock', [])) or 'none read'}")
        files = [x.split(" ", 1)[1] for x in pr.get("map", [])] + pr.get("fd", [])
        p(f"  the live rank's mapped and open files by origin: {origins(files, pr['lib_real'].rsplit('/tools/', 1)[0] if '/tools/' in pr['lib_real'] else '/data/models')}; "
          f"{' '.join(pr.get('rss', []))}")
        for c in pr.get("cite", []):
            p(f"  {c}")
        if pr.get("scratch_exists") != "0":
            fail(f"{ip}: {S} or {CS} exists already")
        if pr.get("own_units") != "0":
            fail(f"{ip}: strixflerp-stall units exist already")
        if pr.get("first_light") != "1":
            fail(f"{ip}: no {TOOLS}/first_light.sh")
        if pr.get("cache_writable") != "1" or int(pr.get("cache_avail_kb") or 0) < 4 << 20:
            fail(f"{ip}: /cache is not writable or has under 4 GiB free (the scratch LEARN_DIR)")
        if "noexec" in (pr.get("cache_mount") or ""):
            p("  /cache is noexec: the libraries of arm L go to the RAM overlay, not /cache")
        if int(pr.get("ram_avail_kb") or 0) < ((staged or 0) >> 10) + (2 << 20):
            fail(f"{ip}: the RAM overlay has no room for the staged libraries + 2 GiB")
        master_if = next((a.split()[0] for a in pr.get("addr", []) if a.split()[1].split("/")[0] == live["MASTER_IP"]), None)
        if rank == 0:
            facts["master_if"] = master_if
            p(f"  MASTER_IP {live['MASTER_IP']} is this node's {master_if or 'NO'} interface ({'rail' if live['MASTER_IP'].startswith(RAIL_NET) else 'NOT the rail'} network)")
            if not master_if:
                fail(f"MASTER_IP {live['MASTER_IP']} is not an address of rank 0's node")
        else:
            r = next((x for x in pr.get("route", []) if x.startswith(live["MASTER_IP"] + " ")), "?")
            facts["master_route"] = r
            p(f"  route from rank 1's node to MASTER_IP: {r}")
        facts.setdefault("lib_real", pr["lib_real"])
        facts.setdefault("libs", {}).update(libs or {})
    facts["lib"], facts["engram"], facts["live"] = lib, engram, live
    gate = int(live["PEAK_GIB"]) + 8
    p(f"\narm L stages about {(staged or 0) / 2**20:.0f} MiB a node on the RAM overlay (the mapped libraries + .kpack, not the 8.5 GiB directory: a rank "
      f"starts only with MemAvailable >= {gate} GiB (serve_lib.sh serve_idle) and an idle node has about 117); arm L is skipped if MemAvailable after staging is under {gate + 1} GiB")
    p(f"arm E: ENGRAM_AIO=1; the Engram shards are not staged ({int((facts.get('rank1') or {}).get('eng_bytes') or 0) / 2**30:.0f} GiB against "
      f"{int((facts.get('rank1') or {}).get('cache_avail_kb') or 0) / 2**20:.0f} GiB free on rank 1's /cache)")
    p("\nthe window:")
    p(f"  OWNER=agent TTL={int(minutes * 60 + 600)} pod-quad hold {F} {A}; KANBAN_ACTOR=agent {LAUNCHER} down; wait for both GPUs idle")
    p(f"  mkdir {S}/{{conf,run,samples}} and {CS} on both workers; the sampler and the arms' confs copied there")
    for arm in arms:
        p(f"  arm {arm}: conf = the serving conf + " + " ".join(f"{k}='{v}'" for k, v in overrides(arm, live).items()))
    p(f"  each arm: sampler units ({SUNIT}, nice 5, every 2 s) on both workers; `RUN_ID=stallx-<arm>-<time> TF_SERVE_CONF={S}/conf/<arm>.conf "
      f"./first_light.sh up` as unit {UNIT} on rank 1's node; the ladder; files collected; first_light.sh down")
    p(f"  ladder: the standard prefix once ({PREFIX}, about {LEARN_S} s, learns its state into the scratch LEARN_DIR), its ladder "
      f"{WL.LADDER} words twice, then a 4,096-token state and its ladder up to 3 times (stops after a pass with a rank 1 frame over {T['stall_ms']} ms)")
    p(f"  time: an arm is about {BOOT_S} + {LEARN_S} + 2 x {STD_PASS_S} + {SMALL_S} + {DOWN_S} s = {(BOOT_S + LEARN_S + 2 * STD_PASS_S + SMALL_S + DOWN_S) / 60:.0f} min "
      f"when nothing stalls; --minutes {minutes:g} is the release time (work stops {TAIL_S} s before it; a short arm drops the standard ladder's passes first)")
    p(f"  at the end, always: stop {UNIT} and {SUNIT}, first_light.sh down, collect to {os.path.relpath(OUT_ROOT, ROOT)}/<UTC>/, rm -rf {S} {CS}, "
      "pod-quad release FIRST, then wait up to 8 min for the keeper and record /health")
    p("thresholds: " + json.dumps(T))
    p(f"\n{'PRECONDITIONS OK' if not bad else 'PRECONDITIONS FAILED: ' + '; '.join(bad)}")
    return not bad, facts


def frames_now(run, rid):
    _, text = run.on(A, f"grep -o 'spent [0-9]* ms on its fill_chunk frame' {S}/run/{rid}/rank1.log 2>/dev/null", 30, True)
    return [int(x) for x in re.findall(r"spent (\d+) ms", text)]


def rank_files(run, ip, rank, rid, lib_real):
    """What the booted rank maps and holds open, and its sockets (read-only)."""
    _, text = run.on(ip, f"p=$(cat {S}/run/{rid}/rank{rank}.pid 2>/dev/null); awk '$6 ~ /^\\// {{print $6}}' /proc/$p/maps | sort -u; "
                         f"ls -l /proc/$p/fd | awk '{{print $NF}}' | grep '^/' | grep -v '^/dev\\|^/proc\\|^/sys\\|^/run' | sort -u; "
                         f"echo ==; ss -tnp 2>/dev/null | grep \"pid=$p,\" | awk '{{print $4 \" \" $5}}'", 30, True)
    files, _, socks = text.partition("==")
    paths = [x for x in files.split("\n") if x.startswith("/")]
    mgmt_root = lib_real.rsplit("/tools/", 1)[0] if "/tools/" in lib_real else "/data/models"
    return {"origins": origins(paths, mgmt_root), "on_mgmt_nfs": sorted(x for x in paths if x.startswith(mgmt_root + "/"))[:40],
            "sockets": [x for x in socks.strip().split("\n") if x]}


def stage_libs(run, facts, gate_gib):
    """Arm L's copy on each worker's RAM overlay: the mapped libraries (tree kept: their RUNPATHs are $ORIGIN
    relative), the symlinks beside them, and .kpack beside lib/. An error string, or None."""
    src = facts["lib_real"].rstrip("/")
    rel = sorted(facts.get("libs") or DEFAULT_LIBS)
    dirs = sorted({os.path.dirname(x) for x in rel})
    script = (f"set -e; rm -rf {S}/therock; mkdir -p {S}/therock/lib; cd {shlex.quote(src)}; "
              f"cp --parents -p {' '.join(shlex.quote(x) for x in rel)} {S}/therock/lib/; "
              + "".join(f"find ./{d} -maxdepth 1 -type l -exec cp -P -t {S}/therock/lib/{d} {{}} + ; " for d in dirs)
              + f"cp -r ../.kpack {S}/therock/.kpack; du -sb {S}/therock | cut -f1; awk '/^MemAvailable:/ {{print $2}}' /proc/meminfo")
    for ip in (F, A):
        rc, text = run.on(ip, script, 300)
        nums = [int(x) for x in text.split() if x.isdigit()]
        if rc != 0 or len(nums) < 2:
            return f"could not stage the libraries on {ip}"
        run.say(f"{ip}: staged {nums[-2] / 2**20:.0f} MiB of libraries, MemAvailable {nums[-1] / 2**20:.1f} GiB")
        if nums[-1] < (gate_gib + 1) << 20:
            return f"{ip}: MemAvailable {nums[-1] / 2**20:.1f} GiB after staging, a rank needs {gate_gib}"
    return None


def clock_offsets(run):
    """Each node's clock minus this host's, seconds."""
    out = {}
    for ip, name in ((F, "rank0"), (A, "rank1")):
        t0 = time.time()
        rc, text = run.on(ip, "date +%s.%N", 20, True)
        t1 = time.time()
        try:
            out[name] = round(float(text.strip().splitlines()[-1]) - (t0 + t1) / 2, 3)
        except (ValueError, IndexError):
            out[name] = None
    return out


def collect(run, tag, rid):
    for ip, r in ((F, 0), (A, 1)):
        for src, dst in ((f"{S}/samples/{tag}-rank{r}.jsonl", f"{tag}-rank{r}.samples.jsonl"), (f"{S}/run/{rid}/rank{r}.log", f"{tag}-rank{r}.log")):
            run.sh(["rsync", "-a", "-e", " ".join(SSH), f"pod@{ip}:{src}", os.path.join(run.out, dst)], 120, True)
    run.sh(["rsync", "-a", "-e", " ".join(SSH), f"pod@{A}:{S}/run/{rid}/first_light.json", os.path.join(run.out, f"{tag}-first_light.json")], 60, True)


def ladder(run, row, base, label, passes, seed0, stop_on_stall, rid):
    """`passes` passes of the warm ladder on `base`; False when the time ran out. Stops after a pass with a stalled
    rank 1 frame when stop_on_stall."""
    for rep in range(1, passes + 1):
        n0 = len(frames_now(run, rid))
        for k in WL.LADDER:
            need = 25 + k / 250.0                    # a stalled 32,000-word append took over 95 s on 2026-11-11
            if run.left() < TAIL_S + DOWN_S + need:
                row["trimmed"].append(f"{label} pass {rep} from {k} words: {int(run.left())} s to the release time")
                run.say("arm " + row["arm"] + ": " + row["trimmed"][-1])
                return False
            t0 = time.time()
            WL.ask(base, k, seed0 * rep + k, row["requests"], label)
            row["requests"][-1].update(t0=round(t0, 3), t1=round(time.time(), 3), rep=rep)
        new = frames_now(run, rid)[n0:]
        row["passes"].append({"ladder": label, "rep": rep, "rank1_frames": len(new), "max_ms": max(new, default=0)})
        run.say(f"arm {row['arm']} {label} pass {rep}: {len(new)} rank 1 frames, longest {max(new, default=0)} ms")
        if max(new, default=0) > T["stall_ms"]:
            row["stalled"] = True
            if stop_on_stall:
                return True
    return True


live_down = [None]   # the `first_light.sh down` command of a run that is up


def arm(run, tag, facts, std_passes, doc):
    import random
    row = {"arm": tag, "requests": [], "passes": [], "trimmed": [], "stalled": False, "started": now()}
    doc["arms"].append(row)                          # at once: a stop at the release time keeps what the arm did
    rid = "stallx-" + tag + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
    conf = f"{S}/conf/{tag}.conf"
    row.update(run_id=rid, clock_offset_s=clock_offsets(run),
               down=f"cd {TOOLS} && RUN_ID={rid} TF_SERVE_CONF={conf} ./first_light.sh down")
    for ip, r in ((F, 0), (A, 1)):
        run.on(ip, XDG + f"mkdir -p {CS}/learned-{tag}; systemctl --user stop {SUNIT} 2>/dev/null; systemd-run --user --collect --unit={SUNIT} "
               f"--property=Nice=5 python3 -I {S}/pair_stall_sampler.py --rank {r} --run {S}/run/{rid} --out {S}/samples/{tag}-rank{r}.jsonl "
               f"--interval 2 --max-seconds {int(max(300, run.left() + 300))}")
    try:
        live_down[0] = row["down"]
        run.on(A, XDG + f"systemd-run --user --collect --unit={UNIT} --property=KillMode=process bash -c "
               + shlex.quote(f"cd {TOOLS} && RUN_ID={rid} TF_SERVE_CONF={conf} ./first_light.sh up"))
        t0, ok = time.time(), False
        while time.time() - t0 < min(300, max(60, run.left() - TAIL_S - DOWN_S)):
            h = run.health()
            active = run.on(A, XDG + f"systemctl --user is-active -q {UNIT}", 20, True)[0] == 0
            if h and not active:
                ok = True
                break
            if not h and not active and time.time() - t0 > 30:
                break
            time.sleep(4)
        row["startup_s"] = round(time.time() - t0, 1)
        if not ok:
            row["error"] = "did not reach health"
            _, t1 = run.on(A, f"tail -n 30 {S}/run/{rid}/rank1.log {S}/run/{rid}/*.txt 2>/dev/null", 30, True)
            _, t0_ = run.on(F, f"tail -n 40 {S}/run/{rid}/rank0.log 2>/dev/null", 30, True)
            run.save(f"{tag}-failed-start.txt", t1 + "\n==== rank 0\n" + t0_ + "\n")
            return row
        row["health"] = run.health()
        row["files"] = {f"rank{r}": rank_files(run, ip, r, rid, facts["lib_real"]) for ip, r in ((F, 0), (A, 1))}
        run.say(f"arm {tag} up in {row['startup_s']} s; files by origin {json.dumps({k: v['origins'] for k, v in row['files'].items()})}; "
                f"rank 1 sockets {row['files']['rank1']['sockets']}")
        # the standard prefix, read whole once: the request itself learns the 67,584-token state into the scratch dir
        with open(PREFIX) as f:
            src = json.load(f)
        std = {k: src[k] for k in ("model", "shared_messages", "messages", "tools") if k in src}
        avail = run.left() - TAIL_S - DOWN_S
        if avail < LEARN_S + SMALL_S / 2:
            std_passes = -1
        while std_passes > 0 and avail < LEARN_S + std_passes * STD_PASS_S + SMALL_S:
            std_passes -= 1
        row["std_passes_planned"] = std_passes
        if std_passes >= 0:
            WL.ask(std, 250, 999, row["requests"], "standard-learn")
            WL.ask(std, 250, 998, row["requests"], "standard-check")
            row["frames_at_ladder"] = len(frames_now(run, rid))
            if (row["requests"][-1].get("cached_tokens") or 0) >= 60000:
                ladder(run, row, std, "standard-cache", std_passes, 1000, False, rid)
            else:
                run.say(f"arm {tag}: the standard state was not resumed (cached {row['requests'][-1].get('cached_tokens')}): a ~20k-token state of our own")
                r = random.Random(11)
                mid = {"model": "dsv41", "messages": [{"role": "system", "content": "You are a terse assistant."},
                       {"role": "user", "content": "Background text: " + " ".join(r.choice(WL.WORDS) for _ in range(19700))},
                       {"role": "assistant", "content": "Understood."}]}
                WL.ask(mid, 250, 6, row["requests"], "mid-cache-seed")
                ladder(run, row, mid, "mid-cache", max(1, std_passes), 3000, False, rid)
        row.setdefault("frames_at_ladder", len(frames_now(run, rid)))
        r = random.Random(7)                         # the ladder script's 4,096-token state, word for word
        small = {"model": "dsv41", "messages": [{"role": "system", "content": "You are a terse assistant."},
                 {"role": "user", "content": "Background text: " + " ".join(r.choice(WL.WORDS) for _ in range(4300))},
                 {"role": "assistant", "content": "Understood."}]}
        if run.left() > TAIL_S + DOWN_S + 30:
            WL.ask(small, 250, 5, row["requests"], "small-cache-seed")
            ladder(run, row, small, "small-cache", 3, 2000, True, rid)
        row["health_after"] = run.health()
    except Stop:
        raise
    except Exception as e:                           # a dead pair is a result, not the end of the window
        row["error"] = repr(e)[:300]
        run.say(f"arm {tag}: {row['error']}")
    finally:
        frames = frames_now(run, rid)
        row["rank1_fill_chunk_ms"] = frames
        row["stalled"] = row["stalled"] or any(x > T["stall_ms"] for x in frames[row.get("frames_at_ladder", len(frames)):])
        for ip in (F, A):
            run.on(ip, XDG + f"systemctl --user stop {SUNIT} 2>/dev/null; true", 60, True)
        collect(run, tag, rid)
        run.on(A, XDG + f"systemctl --user stop {UNIT} 2>/dev/null; " + row["down"], 300)
        live_down[0] = None
        row["idle_after"] = run.idle()
        row["ended"] = now()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25, help="the hold is released by this many minutes from the start")
    ap.add_argument("--arms", default="", help="S, L, E in order, comma separated; default: S, then L if S stalled and time is left")
    ap.add_argument("--std-passes", type=int, default=2, help="passes of the ladder on the standard state (-1: skip the standard state)")
    ap.add_argument("--busy-ok", action="store_true", help="take the pair down although it has requests in flight")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--analyze", metavar="DIR")
    pair_nodes.add_args(ap)
    a = ap.parse_args()
    if a.analyze:
        an = analyze(a.analyze)
        text = render(an)
        print(text, end="")
        try:
            with open(os.path.join(a.analyze, "analysis.json"), "w") as f:
                json.dump(an, f, indent=1)
            with open(os.path.join(a.analyze, "analysis.txt"), "w") as f:
                f.write(text)
        except OSError:
            pass
        return 0
    pair_nodes.bind(pair_nodes.from_args(a), sys.modules[__name__], PP, WL)
    arms = [t for t in a.arms.split(",") if t in ("S", "L", "E")] or ["S"]
    auto_l = not a.arms
    if a.plan:
        return 0 if plan(arms + (["L"] if auto_l else []), a.minutes, busy_ok=a.busy_ok)[0] else 1
    if a.minutes * 60 < MIN_ARM_S + DOWN_S + TAIL_S + 90:
        sys.exit(f"--minutes {a.minutes:g} is too short for one arm (at least {(MIN_ARM_S + DOWN_S + TAIL_S + 90) / 60:.0f}): nothing done")
    t_start = time.time()
    ok, facts = plan(arms + (["L"] if auto_l else []), a.minutes, busy_ok=a.busy_ok)
    if not ok:
        sys.exit("preconditions failed: nothing done")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    out = os.path.join(OUT_ROOT, stamp)
    os.makedirs(out, exist_ok=True)
    run = Window(out, t_start + a.minutes * 60)
    live = facts["live"]
    doc = {"card": CARD, "started": now(), "release_by": datetime.datetime.fromtimestamp(run.deadline, datetime.timezone.utc).strftime("%H:%M:%SZ"),
           "health_before": run.health(), "thresholds": T, "lib": facts["lib"], "lib_real": facts["lib_real"], "engram": facts["engram"],
           "lib_mount": target_of(facts["rank1"].get("lib_mount")), "engram_mount": target_of(facts["rank1"].get("eng_mount")),
           "master_if": facts.get("master_if"), "master_route": facts.get("master_route"), "pair": PAIR, "arms": [], "skipped": []}
    run.save("plan-facts.json", json.dumps(facts, indent=1))

    def flush():
        run.save("result.json", json.dumps(doc, indent=1))

    def on_signal(signum, _frame):
        for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        signal.alarm(0)
        raise Stop("the release time" if signum == signal.SIGALRM else f"signal {signum}")

    for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, on_signal)
    signal.alarm(max(1, int(run.left() - TAIL_S)))
    hold = False             # True from the moment the hold is asked for: from then on the teardown and release run
    try:
        hold = True
        rc, text = run.sh(f"OWNER=agent REASON={shlex.quote(f'strixflerp PAIR-STALL {CARD} diagnostic window (agreed with the pair owner)')} "
                          f"TTL={int(a.minutes * 60 + 600)} pod-quad hold {F} {A}", 900)
        if rc != 0:
            hold = False
            raise RuntimeError(f"pod-quad hold refused: {text[-300:]}")
        doc["hold_taken"] = now()
        run.sh(f"KANBAN_ACTOR=agent {LAUNCHER} down", 400)
        if not run.idle(240):
            raise RuntimeError("the pair did not go idle under the hold")
        with open(LIVE_CONF) as f:
            conf = f.read()
        for tag in ("S", "L", "E"):
            run.save(f"{tag}.conf", conf_text(conf, tag))
        for ip in (F, A):
            rc, _ = run.on(ip, f"mkdir -p {S}/conf {S}/run {S}/samples {CS}")
            for src, dst in [(SAMPLER, f"{S}/pair_stall_sampler.py")] + [(os.path.join(out, f"{t}.conf"), f"{S}/conf/{t}.conf") for t in ("S", "L", "E")]:
                rc |= run.sh(["rsync", "-a", "-e", " ".join(SSH), src, f"pod@{ip}:{dst}"], 120)[0]
            if rc != 0:
                raise RuntimeError(f"could not set the scratch directory up on {ip}")
        todo = list(arms)
        while todo:
            tag = todo.pop(0)
            if run.left() < MIN_ARM_S + DOWN_S + TAIL_S:
                doc["skipped"].append(f"{tag}: {int(run.left())} s to the release time")
                run.say(doc["skipped"][-1])
                continue
            if tag == "L":
                err = stage_libs(run, facts, int(live["PEAK_GIB"]) + 8)
                if err:
                    doc["skipped"].append("L: " + err)
                    run.say(doc["skipped"][-1])
                    for ip in (F, A):
                        run.on(ip, f"rm -rf {S}/therock")
                    continue
            arm(run, tag, facts, a.std_passes, doc)
            flush()
            if tag == "L":
                for ip in (F, A):
                    run.on(ip, f"rm -rf {S}/therock")
            if tag == "S" and auto_l and doc["arms"][-1]["stalled"]:
                todo.append("L")
    except Stop as e:
        doc["stopped_by"] = str(e)
        run.say("stopping: " + str(e))
    except Exception as e:
        doc["error"] = repr(e)[:400]
        run.say("error: " + doc["error"])
    finally:
        # stop anything of this window, remove the scratch, RELEASE FIRST, then let the pod's keeper restore serving
        for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        signal.alarm(0)
        if not hold:
            flush()
            sys.exit(doc.get("error") or doc.get("stopped_by") or "nothing done")
        try:
            run.on(A, XDG + f"systemctl --user stop {UNIT} 2>/dev/null; " + (live_down[0] or "true"), 300)
            for ip in (F, A):
                run.on(ip, XDG + f"systemctl --user stop {SUNIT} 2>/dev/null; for u in $(systemctl --user list-units 'tfp-*stallx*' --state=active "
                       "--no-legend --plain | cut -d' ' -f1); do systemctl --user stop $u; echo stopped $u; done; true", 120)
            run.idle(120)
            for ip in (F, A):
                run.on(ip, f"rm -rf {S} {CS}")
        except Exception as e:
            doc["teardown_error"] = repr(e)[:300]
        rc, text = run.sh("OWNER=agent pod-quad release", 300)
        doc["hold_released"], doc["release_rc"] = now(), rc
        try:
            time.sleep(2)
            doc["held_after"] = {ip: run.sh(["pod-quad", "held", ip], 30)[0] == 0 for ip in (F, A)}
            doc["left_on_workers"] = {ip: run.on(ip, XDG + f"ls -d {S} {CS} 2>&1 | head -2; systemctl --user list-units 'strixflerp-stall*' --no-legend --plain | head", 60)[1] for ip in (F, A)}
            restored, t0 = None, time.time()
            run.say("hold released; waiting for the pod's keeper to bring serving back (no start from here)")
            while time.time() - t0 < 480:
                h = run.health()
                if h and h.get("model") == "dsv41" and h.get("warming") is False:
                    restored = h
                    break
                time.sleep(5)
            doc["health_after"], doc["restored_at"], doc["restore_wait_s"] = restored, now(), round(time.time() - t0)
            if not restored:
                run.say("SERVING DID NOT COME BACK in 8 minutes after the release: tell the pair's owner (nothing was started from here)")
        finally:
            flush()
    try:
        an = analyze(out)
        run.save("analysis.json", json.dumps(an, indent=1))
        run.save("analysis.txt", render(an))
        print(render(an), end="")
    except Exception as e:
        run.say(f"the analysis failed ({e!r}); rerun it: python3 gate/pair_stall_diag.py --analyze {out}")
    try:
        print(json.dumps({k: doc.get(k) for k in ("hold_taken", "hold_released", "release_rc", "health_after", "skipped", "stopped_by", "error")}, indent=1))
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
