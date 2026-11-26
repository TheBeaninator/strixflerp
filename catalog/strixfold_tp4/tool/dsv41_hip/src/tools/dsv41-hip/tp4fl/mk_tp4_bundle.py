#!/usr/bin/env python3
"""mk_tp4_bundle.py TP2_BUNDLE OUT [--dry]: the TP4 inputs bundle's metadata (716f35d40a), after TP4-RANKFILES
(go-tp4-rankfiles.service) cut, checked and PLACEd rank0of4..rank3of4 and pruned them from OUT. Writes only small files
in OUT: relative links model, kit and dsv41_token_map.json into the TP2 bundle (same export, so they resolve on the
workers' rail mount too), and inputs_manifest.json = the TP2 manifest with rank_caches for the four TP4 files (name and
size from their PLACE manifests, the full-byte check from OUT/rank<g>-check.log, which must say ok) plus
derived_from (the TP2 manifest's sha256 and the reslice). Prints the new manifest's sha256 (tp4-fl.conf BUNDLE_SHA256).
Refuses when any check log is missing or not ok, or a PLACE manifest is missing. Moves and copies no weights."""
import glob, hashlib, json, os, sys

def main():
    tp2, out = sys.argv[1:3]; dry = "--dry" in sys.argv
    man = "/data/models/.place/manifests/tp4-rank"
    m = json.load(open(os.path.join(tp2, "inputs_manifest.json")))
    caches = []
    for g in range(4):
        ms = glob.glob(f"{man}/rank{g}of4-*.bin.manifest.json")
        if len(ms) != 1: sys.exit(f"rank {g}: {len(ms)} PLACE manifests in {man}")
        pm = json.load(open(ms[0])); name = pm["file_name"]
        chk = os.path.join(out, f"rank{g}-check.log")
        rep = None
        for line in (open(chk).read().splitlines() if os.path.exists(chk) else []):
            try: j = json.loads(line)
            except Exception: continue
            if isinstance(j, dict) and j.get("file") == name: rep = j
        if not rep or rep.get("ok") is not True: sys.exit(f"rank {g}: {chk} has no ok report for {name}")
        caches.append({"rank": g, "world": 4, "file": f"rank{g}/rank-cache/{name}", "bytes": int(pm["file_size"]),
                       "place_manifest": ms[0], "place_root": pm["root"],
                       "build": {"report": {"file": name, "rank": g, "world": 4}, "tool": "tf-dsv41-rank-reslice"},
                       "byte_check": {"log": f"rank{g}-check.log", "report": rep}})
    raw2 = open(os.path.join(tp2, "inputs_manifest.json"), "rb").read()
    m["rank_caches"] = caches
    m["derived_from"] = {"tp2_manifest_sha256": hashlib.sha256(raw2).hexdigest(), "tp2_bundle": tp2,
                         "how": "rank files cut from the TP2 rank files (TP4-RANKFILES b7eb57e048), card 716f35d40a"}
    s = json.dumps(m, indent=1, sort_keys=True) + "\n"
    sha = hashlib.sha256(s.encode()).hexdigest()
    if dry:
        print(f"DRY: would write {out}/inputs_manifest.json sha256 {sha} and links model kit dsv41_token_map.json"); return
    os.makedirs(out, exist_ok=True)
    rel = os.path.relpath(tp2, out)
    for n in ("model", "kit", "dsv41_token_map.json"):
        p = os.path.join(out, n)
        if os.path.islink(p) and os.readlink(p) == os.path.join(rel, n): continue
        if os.path.lexists(p): sys.exit(f"{p} exists and is not the expected link")
        os.symlink(os.path.join(rel, n), p)
    open(os.path.join(out, "inputs_manifest.json"), "w").write(s)
    print(sha)

main()
