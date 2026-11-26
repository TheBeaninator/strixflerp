#!/usr/bin/env python3
"""The TP2 -> TP4 re-slicer (tf-dsv41-rank-reslice, reslice_tp4.sh) on CPU fixtures at the real shapes, no weights:

1. real schema: the re-slicer's --index-only on the real TP2 bundle (REAL_TP2: index only, nothing written) gives,
   for nodes 0-3, the same index as the Python schema on the sparse real-header fixture;
2. toy: TP2 rank files built from the toy checkpoint by tf-dsv41-rank-cache, then for each node (one at a time) the
   re-sliced file is byte-identical to the file tf-dsv41-rank-cache builds from the checkpoint at world 4, and its
   --check against the TP2 file passes;
3. planted mismatches: one flipped byte in a column-cut trellis, in a whole entry and in the footer's index each fail
   --check (exit 1), and the restored file passes again;
4. refusals: node 4, an unknown flag.

usage: test_reslice4.py INVENTORY_JSON CONFIG_JSON WORKDIR BIN_DIR REAL_TP2_BUNDLE
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory_checkpoint import rank_plan, scan  # noqa: E402


def run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, **kw)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def flip(path: Path, at: int) -> None:
    with path.open("r+b") as f:
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0x5A]))


def report(p) -> dict:
    lines = [x for x in p.stdout.decode().splitlines() if x.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def main() -> int:
    inventory, config, work, bindir, real = (Path(x) for x in sys.argv[1:6])
    rank_cache, reslice = bindir / "tf-dsv41-rank-cache", bindir / "tf-dsv41-rank-reslice"
    failures, summary = [], {}
    for mode in ("sparse", "toy"):
        if not (work / mode).exists():
            subprocess.run([sys.executable, HERE / "rank_fixture.py", inventory, config, work / mode, "--" + mode], check=True)

    # 1. the real TP2 files: the re-slicer's index == the Python schema's
    raw = json.loads((work / "sparse" / "config.json").read_text())
    tensors, _ = scan(work / "sparse")
    for g in range(4):
        plan = rank_plan(raw, tensors, g, 4)
        p = run([reslice, real, work / "unused", g, "--index-only"])
        got = hashlib.sha256(p.stdout).hexdigest()
        ok = p.returncode == 0 and got == plan["index_sha256"]
        summary[f"real TP2 -> rank{g}of4 index"] = {"sha20": got[:20], "rank_file_bytes": plan["rank_file_bytes"], "equals_python": ok}
        if not ok:
            failures.append(f"real node {g}: index {got[:20]} rc {p.returncode} != python {plan['index_sha256'][:20]}")

    # 2. toy TP2 bundle, then node by node
    toy = work / "toy"
    raw = json.loads((toy / "config.json").read_text())
    tensors, _ = scan(toy)
    tp2 = work / "toy-tp2"
    if tp2.exists():
        raise SystemExit(f"{tp2} exists: remove it first")
    (tp2 / "model").mkdir(parents=True)
    shutil.copyfile(toy / "config.json", tp2 / "model" / "config.json")
    for r in range(2):
        d = tp2 / f"rank{r}" / "rank-cache"
        d.mkdir(parents=True)
        p = run([rank_cache, toy, d, r, 2])
        if p.returncode:
            raise SystemExit(f"toy TP2 rank {r} build failed: {p.stdout[-500:]!r}")
    out, direct = work / "toy-tp4", work / "toy-direct"
    for g in range(4):
        plan = rank_plan(raw, tensors, g, 4)
        p = run(["bash", HERE / "reslice_tp4.sh", tp2, out, g], env=os.environ | {"TFP_BIN": str(bindir), "FLOOR_GB": "0"})
        if p.returncode:
            failures.append(f"node {g}: reslice_tp4.sh rc {p.returncode}: {p.stderr.decode()[-300:]}")
            continue
        d = direct / f"rank{g}"
        d.mkdir(parents=True)
        if run([rank_cache, toy, d, g, 4]).returncode:
            failures.append(f"node {g}: direct build failed")
            continue
        (cut_file,) = (out / f"rank{g}" / "rank-cache").glob(f"rank{g}of4-*.bin")
        (direct_file,) = d.glob(f"rank{g}of4-*.bin")
        identical = cut_file.name == direct_file.name and sha(cut_file) == sha(direct_file)
        chk = json.loads((out / f"rank{g}-check.log").read_text().splitlines()[-1])
        row = {"file": cut_file.name, "bytes": cut_file.stat().st_size, "identical_to_checkpoint_build": identical,
               "check_ok": chk["ok"], "compared": chk["compared"], "compared_bytes": chk["compared_bytes"],
               "byte_mismatches": chk["byte_mismatches"]}
        if not identical or chk["ok"] is not True or chk["compared"] != plan["entry_count"] or chk["compared_bytes"] != plan["data_bytes"]:
            failures.append(f"node {g}: {row}")
        # 3. planted mismatches: a column-cut trellis byte, a whole-entry byte, an index byte
        entries = {e[0]: e for e in plan["entries"]}
        cut_tr = next(e for k, e in entries.items() if ".ffn.experts.0.w1|" in k and k.endswith("|tr"))
        whole = entries["norm.weight|None"]
        index_at = plan["data_bytes"] + 10
        planted = {}
        for label, at in (("column-cut trellis", cut_tr[3] + cut_tr[4] // 2 + 1), ("whole entry", whole[3] + 3), ("index", index_at)):
            flip(cut_file, at)
            q = run([reslice, tp2, cut_file.parent, g, "--check"])
            planted[label] = q.returncode
            if q.returncode != 1:
                failures.append(f"node {g}: planted {label} mismatch not caught (rc {q.returncode})")
            flip(cut_file, at)
        q = run([reslice, tp2, cut_file.parent, g, "--check"])
        row["planted_rc"] = planted
        row["restored_check_rc"] = q.returncode
        if q.returncode != 0:
            failures.append(f"node {g}: restored file fails --check")
        summary[f"toy rank{g}of4"] = row
        # one node at a time: this node's files go before the next is built
        shutil.rmtree(out / f"rank{g}")
        shutil.rmtree(d)

    # 4. refusals
    for args in ([4], [0, "--bogus"]):
        p = run([reslice, tp2, work / "unused", *args])
        summary[f"refuse {args}"] = p.returncode
        if p.returncode != 2:
            failures.append(f"accepted {args}")
    print(json.dumps(summary, indent=1))
    print("FAIL" if failures else "PASS", *failures, sep="\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
