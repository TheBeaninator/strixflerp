#!/usr/bin/env python3
"""A domain's Edge-REAP snapshots turned into a MASK_EXPERTS file of the experts to remove (pod card 97f0f92e7a, KCUT).

    mask_keepset.py --snap-dir SNAPSHOTS --domain code --keep 0.5 --out code-k50.mask --report code-k50.json
    mask_keepset.py --report-in code.text.json --keep 0.375 --out code-k375.mask      (explicit files, no hubs from others)

The rules, in the order they apply, per MoE layer (the report's layers; the MTP blocks and the shared expert are not in
a report and are never named in the mask):
  1. budget k = round(keep * experts), never below FLOOR (6, the router's top_k: the engine refuses a layer left with
     fewer than 6 experts);
  2. hubs: an expert whose share of the layer's gate mass is at least --hub-share (default 0.02, about 5x the uniform
     1/384) in ANY report given (the domain's own and every --hub-report / every other domain in --snap-dir) is kept,
     most-shared first; if the hubs alone exceed k the k most-shared are kept (the budget and floor win; said in the
     report as hubs_dropped);
  3. the rest of the budget goes to the experts of the highest REAP saliency in the layer (reap_saliency of the merged
     domain report, the mean of gate x expert-output norm over the tokens routed to it; with --count-tau T > 0 it is
     damped by 1 - exp(-count / T) as owner's Edge-REAP score is); ties go to the lower expert id;
  4. everything else is removed. A layer with nothing removed writes no line.

The output is the engine's format: '#' comments, one 'layer:e1,e2,...' line a layer (removed experts, ascending).
The report (--report) gives, per layer and overall, how much of the domain's saliency mass (reap_weighted_norm_sum),
traffic (token_count) and gate mass the kept experts hold, with the hub and floor accounting.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import edge_reap_report as R  # noqa: E402

FLOOR = 6
HUB_SHARE = 0.02
HUB_MIN_TOKENS = 20000  # a report with fewer tokens is too thin to name hubs (13 tokens put 6 experts at share 1/6)
TARGET_LAYERS = 40  # DeepSeek-V4.1-Flash: the MTP blocks are past it and never in a report


class KeepsetError(ValueError):
    pass


def budget(keep: float, experts: int) -> int:
    if not (0.0 < keep <= 1.0) or not math.isfinite(keep):
        raise KeepsetError(f"keep must be in (0, 1], got {keep!r}")
    return min(experts, max(FLOOR, int(math.floor(keep * experts + 0.5))))


def gate_shares(rep: Mapping[str, Any]) -> List[List[float]]:
    """Per layer, each expert's share of the layer's gate mass in one report (0 for a layer with none)."""
    out = []
    for row in rep["gate_mass_sum"]:
        tot = float(sum(row))
        out.append([float(v) / tot if tot > 0 else 0.0 for v in row])
    return out


def hub_scores(hub_reports: Sequence[Mapping[str, Any]], layers: int, experts: int) -> List[List[float]]:
    """[layer][expert]: the largest gate-mass share the expert holds in any of the reports."""
    best = [[0.0] * experts for _ in range(layers)]
    for rep in hub_reports:
        if (rep["num_layers"], rep["num_experts"]) != (layers, experts):
            raise KeepsetError("a hub report has another shape than the domain report")
        for l, row in enumerate(gate_shares(rep)):
            for j, v in enumerate(row):
                if v > best[l][j]:
                    best[l][j] = v
    return best


def solve(domain: Mapping[str, Any], keep: float, hub_reports: Sequence[Mapping[str, Any]] = (),
          hub_share: float = HUB_SHARE, count_tau: float = 0.0) -> Dict[str, Any]:
    """The keep-set: {'kept': {layer: sorted ids}, 'hubs': {layer: sorted ids kept as hubs}, 'hubs_dropped': {layer: n},
    'budget': k}. Pure and deterministic."""
    dom = R.validate(domain)
    L, E = dom["num_layers"], dom["num_experts"]
    k = budget(keep, E)
    sal, cnt = dom["reap_saliency"], dom["token_count"]
    share = hub_scores(list(hub_reports) + [dom], L, E)
    kept: Dict[int, List[int]] = {}
    hubs: Dict[int, List[int]] = {}
    dropped: Dict[int, int] = {}
    for l in range(L):
        cand = sorted((j for j in range(E) if share[l][j] >= hub_share and share[l][j] > 0.0),
                      key=lambda j: (-share[l][j], j))
        dropped[l] = max(0, len(cand) - k)
        hub = cand[:k]
        score = [float(sal[l][j]) * ((1.0 - math.exp(-cnt[l][j] / count_tau)) if count_tau > 0 else 1.0)
                 for j in range(E)]
        chosen = set(hub)
        for j in sorted(range(E), key=lambda j: (-score[j], j)):
            if len(chosen) >= k:
                break
            chosen.add(j)
        kept[l] = sorted(chosen)
        hubs[l] = sorted(hub)
    return {"kept": kept, "hubs": hubs, "hubs_dropped": dropped, "budget": k}


def removed(kept: Mapping[int, Sequence[int]], experts: int) -> Dict[int, List[int]]:
    out = {}
    for l, ids in kept.items():
        keep = set(ids)
        r = [j for j in range(experts) if j not in keep]
        if r:
            out[l] = r
    return out


def canonical(rem: Mapping[int, Sequence[int]]) -> str:
    """The engine's canonical text of a mask (its id is the first 16 hex of the SHA-256 of this)."""
    return "".join("%d:%s\n" % (l, ",".join(str(e) for e in sorted(rem[l]))) for l in sorted(rem) if rem[l])


def mask_id(rem: Mapping[int, Sequence[int]]) -> str:
    c = canonical(rem)
    return hashlib.sha256(c.encode()).hexdigest()[:16] if c else ""


def mask_text(rem: Mapping[int, Sequence[int]], header: Sequence[str]) -> str:
    lines = ["# " + h for h in header]
    lines += ["%d:%s" % (l, ",".join(str(e) for e in rem[l])) for l in sorted(rem)]
    return "\n".join(lines) + "\n"


def coverage(domain: Mapping[str, Any], kept: Mapping[int, Sequence[int]]) -> Dict[str, Any]:
    """Per layer and overall: the share of saliency mass, traffic and gate mass the kept experts hold."""
    rep = R.validate(domain)
    grids = {"saliency": rep["reap_weighted_norm_sum"], "traffic": rep["token_count"],
             "gate_mass": rep["gate_mass_sum"]}
    per_layer = []
    tot = {n: [0.0, 0.0] for n in grids}
    for l in range(rep["num_layers"]):
        row = {"layer": l, "kept": len(kept[l])}
        for n, g in grids.items():
            all_ = float(sum(g[l]))
            got = float(sum(g[l][j] for j in kept[l]))
            row[n] = got / all_ if all_ > 0 else None
            tot[n][0] += got
            tot[n][1] += all_
        per_layer.append(row)
    overall = {n: (a / b if b > 0 else None) for n, (a, b) in tot.items()}
    return {"per_layer": per_layer, "overall": overall,
            "min": {n: min((r[n] for r in per_layer if r[n] is not None), default=None) for n in grids}}


def load_reports(paths: Sequence[str]) -> List[Dict[str, Any]]:
    if not paths:
        raise KeepsetError("no report files")
    return [R.load(p) for p in paths]


def snap_files(snap_dir: str, domain: str, modalities: Optional[Sequence[str]] = None):
    """(the domain's files, the other domains' files) from SNAP_DIR/<domain>.<modality>.json."""
    mine, others = [], []
    for p in sorted(glob.glob(os.path.join(snap_dir, "*.json"))):
        parts = os.path.basename(p)[:-5].rsplit(".", 1)
        if len(parts) != 2:
            continue
        dom, mod = parts
        if dom == domain:
            if not modalities or mod in modalities:
                mine.append(p)
        else:
            others.append(p)
    if not mine:
        raise KeepsetError(f"no snapshot for domain {domain!r} in {snap_dir}")
    return mine, others


def report_tokens(rep: Mapping[str, Any], top_k: int = FLOOR) -> int:
    """Tokens a report saw: its 'tokens' field, else the routed count / top_k."""
    t = rep.get("tokens")
    if isinstance(t, int) and not isinstance(t, bool) and t > 0:
        return t
    return int(sum(sum(r) for r in rep["token_count"]) // top_k)


def build(domain_paths: Sequence[str], keep: float, hub_paths: Sequence[str] = (), hub_share: float = HUB_SHARE,
          count_tau: float = 0.0, label: str = "", hub_min_tokens: int = HUB_MIN_TOKENS) -> Dict[str, Any]:
    reps = load_reports(domain_paths)
    dom = reps[0] if len(reps) == 1 else R.merge(reps)
    loaded = [(p, R.load(p)) for p in hub_paths]
    hubs = [r for p, r in loaded if report_tokens(r) >= hub_min_tokens]
    skipped = [os.path.basename(p) for p, r in loaded if report_tokens(r) < hub_min_tokens]
    sol = solve(dom, keep, hubs, hub_share, count_tau)
    E = dom["num_experts"]
    rem = removed(sol["kept"], E)
    cov = coverage(dom, sol["kept"])
    nh = [len(sol["hubs"][l]) for l in sorted(sol["hubs"])]
    info = {"domain": label, "keep": keep, "keep_experts": sol["budget"], "experts": E, "layers": dom["num_layers"],
            "floor": FLOOR, "hub_share": hub_share, "count_tau": count_tau,
            "domain_files": [os.path.basename(p) for p in domain_paths],
            "hub_files": [os.path.basename(p) for p, r in loaded if report_tokens(r) >= hub_min_tokens],
            "hub_files_skipped_thin": skipped, "hub_min_tokens": hub_min_tokens,
            "domain_tokens": sum(report_tokens(r) for r in reps),
            "tokens_routed": int(sum(sum(r) for r in dom["token_count"])),
            "removed_total": sum(len(v) for v in rem.values()),
            "removed_per_layer": [len(rem.get(l, [])) for l in range(dom["num_layers"])],
            "hubs_per_layer": nh, "hubs_dropped_layers": sorted(l for l, n in sol["hubs_dropped"].items() if n),
            "mask_id": mask_id(rem), "coverage": cov}
    header = [f"MASK_EXPERTS keep-set for domain {label or '?'}: keep {keep:g} = {sol['budget']} of {E} experts per layer "
              f"({dom['num_layers']} layers), the removed experts listed",
              f"rules: saliency rank per layer, floor {FLOOR}, hubs (gate-mass share >= {hub_share:g} in any of "
              f"{len(hubs) + 1} reports with >= {hub_min_tokens} tokens) kept, MTP layers and the shared expert never named",
              f"id {info['mask_id']}; saliency coverage {cov['overall']['saliency']:.4f}, traffic coverage "
              f"{cov['overall']['traffic']:.4f}"]
    return {"text": mask_text(rem, header), "removed": rem, "info": info}


def report_md(info: Mapping[str, Any]) -> str:
    c = info["coverage"]
    out = [f"# keep-set {info['domain']} keep {info['keep']:g}: {info['keep_experts']} of {info['experts']} per layer",
           "", f"mask id {info['mask_id']}; removed {info['removed_total']} (layer, expert) cells; "
           f"{info['tokens_routed']} routed tokens in the domain report; hub share >= {info['hub_share']:g}, "
           f"floor {info['floor']}.", "",
           "| coverage of | overall | worst layer |", "|---|---|---|"]
    for n in ("saliency", "traffic", "gate_mass"):
        out.append(f"| {n} | {c['overall'][n]:.4f} | {c['min'][n]:.4f} |")
    out += ["", "| layer | kept | hubs | saliency | traffic | gate mass |", "|---|---|---|---|---|---|"]
    for r, h in zip(c["per_layer"], info["hubs_per_layer"]):
        f = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        out.append(f"| {r['layer']} | {r['kept']} | {h} | {f(r['saliency'])} | {f(r['traffic'])} | {f(r['gate_mass'])} |")
    return "\n".join(out) + "\n"


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def read_mask(path: str, experts: int = 384) -> Dict[int, List[int]]:
    """A MASK_EXPERTS file's removed experts, {layer: ids} (comments skipped; no engine refusals, build() makes valid ones)."""
    rem: Dict[int, set] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                head, tail = line.split(":", 1)
                rem.setdefault(int(head), set()).update(int(x) for x in tail.split(","))
    return {l: sorted(v) for l, v in rem.items()}


def kept_of(rem: Mapping[int, Sequence[int]], layers: int, experts: int) -> Dict[int, List[int]]:
    return {l: [j for j in range(experts) if j not in set(rem.get(l, ()))] for l in range(layers)}


def overlap(mask_a: str, mask_b: str, cover_report: Optional[str] = None, layers: int = TARGET_LAYERS,
            experts: int = 384) -> Dict[str, Any]:
    """Jaccard of the two masks' KEPT sets per layer (median and range), and, with cover_report, the share of that
    report's saliency, traffic and gate mass each mask's kept set holds."""
    ka = kept_of(read_mask(mask_a), layers, experts)
    kb = kept_of(read_mask(mask_b), layers, experts)
    jac = []
    for l in range(layers):
        a, b = set(ka[l]), set(kb[l])
        jac.append(len(a & b) / len(a | b))
    s = sorted(jac)
    med = (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2 if len(s) % 2 == 0 else s[len(s) // 2]
    out: Dict[str, Any] = {"a": os.path.basename(mask_a), "b": os.path.basename(mask_b), "jaccard_per_layer": jac,
                           "jaccard_median": med, "jaccard_min": s[0], "jaccard_max": s[-1]}
    if cover_report:
        rep = R.load(cover_report)
        out["cover_report"] = os.path.basename(cover_report)
        out["coverage_overall"] = {os.path.basename(p): coverage(rep, ks)["overall"] for p, ks in ((mask_a, ka), (mask_b, kb))}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "overlap":
        op = argparse.ArgumentParser(prog="mask_keepset.py overlap")
        op.add_argument("mask_a")
        op.add_argument("mask_b")
        op.add_argument("--cover", help="a report whose saliency/traffic each mask's kept set is scored against")
        op.add_argument("--out")
        oa = op.parse_args(argv[1:])
        try:
            res = overlap(oa.mask_a, oa.mask_b, oa.cover)
        except (R.ReportError, OSError, ValueError) as e:
            print(f"mask_keepset: {e}", file=sys.stderr)
            return 2
        txt = json.dumps(res, indent=1) + "\n"
        if oa.out:
            _write(oa.out, txt)
        print(f"jaccard of kept sets: median {res['jaccard_median']:.4f} (min {res['jaccard_min']:.4f}, max {res['jaccard_max']:.4f})")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snap-dir", help="directory of <domain>.<modality>.json snapshots")
    ap.add_argument("--domain", help="the domain to keep (with --snap-dir); the other domains there are hub sources")
    ap.add_argument("--modality", action="append", default=[], help="only these modalities of the domain (default all)")
    ap.add_argument("--report-in", action="append", default=[], help="explicit domain report file (merged if several)")
    ap.add_argument("--hub-report", action="append", default=[], help="extra report whose hubs are always kept")
    ap.add_argument("--keep", type=float, required=True, help="fraction of experts kept per layer, e.g. 0.5 or 0.375")
    ap.add_argument("--hub-share", type=float, default=HUB_SHARE)
    ap.add_argument("--hub-min-tokens", type=int, default=HUB_MIN_TOKENS,
                    help="hub reports with fewer tokens are skipped (too thin to name hubs)")
    ap.add_argument("--count-tau", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="the MASK_EXPERTS file to write")
    ap.add_argument("--report", help="write the coverage report here (.json; a .md beside it)")
    a = ap.parse_args(argv)
    try:
        hubs = list(a.hub_report)
        if a.snap_dir:
            if not a.domain:
                raise KeepsetError("--snap-dir needs --domain")
            mine, others = snap_files(a.snap_dir, a.domain, a.modality)
            dom_paths, hubs = mine + list(a.report_in), hubs + others
        else:
            dom_paths = list(a.report_in)
        res = build(dom_paths, a.keep, hubs, a.hub_share, a.count_tau, a.domain or "", a.hub_min_tokens)
    except (KeepsetError, R.ReportError, OSError, json.JSONDecodeError) as e:
        print(f"mask_keepset: {e}", file=sys.stderr)
        return 2
    _write(a.out, res["text"])
    if a.report:
        _write(a.report, json.dumps(res["info"], indent=1) + "\n")
        _write(os.path.splitext(a.report)[0] + ".md", report_md(res["info"]))
    c = res["info"]["coverage"]["overall"]
    print(f"{a.out}: keep {a.keep:g} = {res['info']['keep_experts']}/{res['info']['experts']} per layer, id "
          f"{res['info']['mask_id']}, saliency {c['saliency']:.4f} traffic {c['traffic']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
