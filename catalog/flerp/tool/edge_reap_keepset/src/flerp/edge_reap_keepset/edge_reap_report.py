#!/usr/bin/env python3
"""Edge-REAP reports from the serving engine's passive collector, read, merged and turned into keep-sets.

The engine (StrixFold DeepSeek-V4.1 with REAP_STATS=1) writes one report per (domain, modality) bucket, from
GET /stats/reap or as a REAP_DIR snapshot, in the format of strix_moe_surgery.py's
DistributedEdgeREAPCollector.export_report plus bookkeeping fields. This module reads such files, merges several
(modalities of one domain, domains into a core report, snapshots of several servers), and calls the script's
compute_edge_reap_keepset on the result. The keep-set solver itself is owner's, unchanged, in strix_moe_surgery.py.

    edge_reap_report.py fetch --url http://HOST:PORT --key-file KEY --domain code [--modality all] --out code.json
    edge_reap_report.py merge REPORT... --out merged.json
    edge_reap_report.py keepset --report REPORT [--report ...] --keep-experts 192 [--core-report R ...] --out keep.json

A merged report is a valid --domain-report for `strix_moe_surgery.py slice`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # keep the catalogue's src/ free of __pycache__
sys.path.insert(0, HERE)
import strix_moe_surgery as surgery  # noqa: E402  (owner's script: the solver and the collector's format)

# export_report's keys, in its order
REPORT_KEYS = ("model_id", "num_layers", "num_experts", "reap_saliency", "reap_weighted_norm_sum", "gate_mass_sum",
               "token_count", "max_l2_norm", "early_pos_count")
SUM_KEYS = ("gate_mass_sum", "token_count", "early_pos_count")
INT_KEYS = ("token_count", "early_pos_count")


class ReportError(ValueError):
    pass


def _grid(rep: Mapping[str, Any], key: str, layers: int, experts: int) -> List[List[float]]:
    g = rep.get(key)
    if not isinstance(g, list) or len(g) != layers:
        raise ReportError(f"{key}: expected {layers} layers")
    for l, row in enumerate(g):
        if not isinstance(row, list) or len(row) != experts:
            raise ReportError(f"{key}[{l}]: expected {experts} experts")
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ReportError(f"{key}[{l}]: not a finite non-negative number: {v!r}")
    return g


def validate(rep: Mapping[str, Any]) -> Dict[str, Any]:
    """Checks a report's keys and shapes; returns it as a plain dict. Extra keys (domain, modality, tokens,
    norm_rows, sampled, build) are kept."""
    missing = [k for k in REPORT_KEYS if k not in rep]
    if missing:
        raise ReportError(f"missing keys: {', '.join(missing)}")
    layers, experts = rep["num_layers"], rep["num_experts"]
    if not isinstance(layers, int) or not isinstance(experts, int) or layers <= 0 or experts <= 0:
        raise ReportError("num_layers / num_experts must be positive integers")
    for k in REPORT_KEYS[3:]:
        _grid(rep, k, layers, experts)
    for k in INT_KEYS:
        if any(not isinstance(v, int) for row in rep[k] for v in row):
            raise ReportError(f"{k}: counts must be integers")
    return dict(rep)


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return validate(json.load(f))


def _sampled(rep: Mapping[str, Any]) -> Optional[Dict[str, List[List[float]]]]:
    s = rep.get("sampled")
    if not isinstance(s, Mapping):
        return None
    nc, ns = s.get("norm_count"), s.get("reap_weighted_norm_sum")
    if nc is None or ns is None:
        return None
    L, E = rep["num_layers"], rep["num_experts"]
    return {"norm_count": _grid(s, "norm_count", L, E), "reap_weighted_norm_sum": _grid(s, "reap_weighted_norm_sum", L, E)}


def merge(reports: Sequence[Mapping[str, Any]], model_id: Optional[str] = None) -> Dict[str, Any]:
    """Adds reports of the same model: counts and gate mass summed, max_l2_norm the maximum. Saliency is the mean
    of g * ||f|| over the sampled (token, expert) pairs when every report carries its raw sampled sums (the
    engine's), else the script's reap_weighted_norm_sum / token_count; reap_weighted_norm_sum is then
    saliency * token_count, so the script's identity holds on the merged report too."""
    if not reports:
        raise ReportError("nothing to merge")
    reps = [validate(r) for r in reports]
    L, E = reps[0]["num_layers"], reps[0]["num_experts"]
    for r in reps[1:]:
        if (r["num_layers"], r["num_experts"]) != (L, E):
            raise ReportError("reports of different shapes")
    if model_id is None:
        ids = {r["model_id"] for r in reps}
        if len(ids) != 1:
            raise ReportError(f"reports of different models: {sorted(ids)}")
        model_id = reps[0]["model_id"]
    zeros = lambda: [[0] * E for _ in range(L)]  # noqa: E731
    out: Dict[str, Any] = {k: zeros() for k in SUM_KEYS}
    out["max_l2_norm"] = [[0.0] * E for _ in range(L)]
    samp = [_sampled(r) for r in reps]
    use_sampled = all(s is not None for s in samp)
    nsum, ncount, wsum = [[0.0] * E for _ in range(L)], zeros(), [[0.0] * E for _ in range(L)]
    for r, s in zip(reps, samp):
        for l in range(L):
            for j in range(E):
                for k in SUM_KEYS:
                    out[k][l][j] += r[k][l][j]
                out["max_l2_norm"][l][j] = max(out["max_l2_norm"][l][j], float(r["max_l2_norm"][l][j]))
                wsum[l][j] += float(r["reap_weighted_norm_sum"][l][j])
                if use_sampled:
                    nsum[l][j] += float(s["reap_weighted_norm_sum"][l][j])
                    ncount[l][j] += s["norm_count"][l][j]
    sal = [[0.0] * E for _ in range(L)]
    wns = [[0.0] * E for _ in range(L)]
    for l in range(L):
        for j in range(E):
            tc = out["token_count"][l][j]
            if use_sampled:
                sal[l][j] = nsum[l][j] / max(ncount[l][j], 1)
                wns[l][j] = sal[l][j] * tc
            else:
                wns[l][j] = wsum[l][j]
                sal[l][j] = wns[l][j] / max(tc, 1)
    rep = {"model_id": model_id, "num_layers": L, "num_experts": E, "reap_saliency": sal,
           "reap_weighted_norm_sum": wns, "gate_mass_sum": out["gate_mass_sum"], "token_count": out["token_count"],
           "max_l2_norm": out["max_l2_norm"], "early_pos_count": out["early_pos_count"]}
    if use_sampled:
        rep["sampled"] = {"norm_count": ncount, "reap_weighted_norm_sum": nsum}
    rep["merged_from"] = [{k: r.get(k) for k in ("domain", "modality", "tokens", "build") if k in r} for r in reps]
    return rep


def keepset(domain_report: Mapping[str, Any], keep_experts: int, core_report: Optional[Mapping[str, Any]] = None,
            **kw: Any) -> Dict[int, List[int]]:
    """owner's compute_edge_reap_keepset on a validated report (same keyword arguments)."""
    return surgery.compute_edge_reap_keepset(validate(domain_report), keep_experts,
                                             core_report=validate(core_report) if core_report else None, **kw)


def fetch(url: str, key: str, domain: str, modality: str = "all", timeout: float = 30.0) -> Dict[str, Any]:
    """GET <url>/stats/reap?domain=..&modality=.. with the server's admin (cache transfer) key."""
    q = urllib.parse.urlencode({"domain": domain, "modality": modality})
    req = urllib.request.Request(url.rstrip("/") + "/stats/reap?" + q, headers={"x-cache-transfer-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return validate(json.loads(r.read().decode("utf-8")))


def _write(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download one bucket's report from a running server")
    f.add_argument("--url", required=True)
    f.add_argument("--key-file", required=True, help="the server's --cache-transfer-key file")
    f.add_argument("--domain", default="default")
    f.add_argument("--modality", default="all", choices=("text", "vision", "all"))
    f.add_argument("--out", required=True)
    m = sub.add_parser("merge", help="add reports (modalities, domains, servers) into one script-compatible report")
    m.add_argument("reports", nargs="+")
    m.add_argument("--model-id", default=None)
    m.add_argument("--out", required=True)
    k = sub.add_parser("keepset", help="compute_edge_reap_keepset on one or more merged reports")
    k.add_argument("--report", action="append", required=True)
    k.add_argument("--core-report", action="append", default=[])
    k.add_argument("--keep-experts", type=int, required=True)
    k.add_argument("--pin-core", type=int, default=16)
    k.add_argument("--pin-super", type=int, default=4)
    k.add_argument("--pin-early-pos", type=int, default=4)
    k.add_argument("--tau-domain", type=float, default=64.0)
    k.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        with open(a.key_file, encoding="utf-8") as fh:
            key = fh.read().strip()
        _write(a.out, fetch(a.url, key, a.domain, a.modality))
    elif a.cmd == "merge":
        _write(a.out, merge([load(p) for p in a.reports], a.model_id))
    else:
        dom = merge([load(p) for p in a.report])
        core = merge([load(p) for p in a.core_report]) if a.core_report else None
        ks = keepset(dom, a.keep_experts, core, pin_core_experts=a.pin_core, pin_super_experts=a.pin_super,
                     pin_early_pos_experts=a.pin_early_pos, tau_domain=a.tau_domain)
        _write(a.out, {"model_id": dom["model_id"], "keep_experts": a.keep_experts,
                       "keepset": {str(l): ids for l, ids in sorted(ks.items())}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
