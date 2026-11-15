#!/usr/bin/env python3
"""Stage the Ornith 397B rank slices for the window engine, outside any window, through PLACE only (no cp/dd/rsync of
weights; the source is the swarm origin on control-node, PLACE's own staging source). Run on control-node as pod.

    stage.py [--dry-run] [--until +12h] [--card ID]

  ranks 0-1 -> control-node /data/models/placed/ornith-w4-r{0,1} (control target; the window streams them over the rail
               through a loom-swarm origin engine.py starts), place request ornith-w4-rR-control
  ranks 2-3 -> the A and C nodes' /cache/flerp-p5 (+ blk.60 and token_embd on C for the MTP head), WORLD-4 slices only
               (place requests ornith-eng-A / ornith-eng-C; the P5 world-3 slice of A is not wanted here)
Nothing is staged on F or D (their /cache holds engram-1017) or B. A node that someone else holds is deferred by PLACE.
--dry-run prints sizes and the commands and writes nothing."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "gate"))
sys.path.insert(0, ROOT)
import ornith_p5 as P  # noqa: E402

GGUF = "Ornith-1.5-397B-Q4_K_M.gguf"


def ranges_for(cfg, rank, mtp=False):
    g = cfg["model"]["gguf"]
    cuts = P.MP.py_plan(g, 4)["cuts"]
    h = P.C.read_header(g)
    spans = {(e[3], e[3] + e[2]) for e in P.C.stage_plan(h, 1, (cuts[rank], cuts[rank + 1]))["ranks"][0]["entries"]}
    if mtp:
        spans |= P.mtp_spans(cfg)
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return (cuts[rank], cuts[rank + 1]), merged


# rank -> (place node, request id, dest dir)
TARGETS = {0: ("control", "ornith-w4-r0-control", "/data/models/placed/ornith-w4-r0"),
           1: ("control", "ornith-w4-r1-control", "/data/models/placed/ornith-w4-r1"),
           2: ("A", "ornith-eng-A", "/cache/flerp-p5"),
           3: ("C", "ornith-eng-C", "/cache/flerp-p5")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until", default="+12h")
    ap.add_argument("--card", default="83a59b0e07")
    ap.add_argument("--ranks", default="0,1,2,3")
    a = ap.parse_args()
    cfg = P.load_cfg()
    s, pc, rc_all = cfg["swarm"], P.place_cmd(), 0
    for r in (int(x) for x in a.ranks.split(",")):
        node, rid, dest = TARGETS[r]
        blocks, ranges = ranges_for(cfg, r, mtp=(r == 3))
        need = sum(e - o for o, e in ranges)
        rf = os.path.join(P.PLACE_RANGES, f"{rid}.ranges")
        print(f"rank {r}: blocks {blocks}{' + MTP block and token_embd' if r == 3 else ''}, {need / P.GIB:.1f} GiB -> {node}:{dest}/{GGUF}.staged by place request {rid}")
        req = pc + ["request", "--id", rid, "--nodes", node, "--dest", dest, "--out", f"{GGUF}.staged", "--manifest", s["manifest"],
                    "--from", s["origin"], "--ranges", f"{node}={rf}", "--until", a.until, "--card", a.card, "--owner", "agent"]
        print("  +", " ".join(req))
        if a.dry_run:
            continue
        os.makedirs(P.PLACE_RANGES, exist_ok=True)
        with open(rf, "w") as fo:
            for o, e in ranges:
                fo.write(f"{o} {e - o}\n")
        rc = subprocess.run(req).returncode or subprocess.run(pc + ["run", "--id", rid]).returncode or subprocess.run(pc + ["verify", "--id", rid]).returncode
        rc_all = rc_all or rc
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
