#!/usr/bin/env python3
"""Join eval verdicts to route records and rank experts whose gate mass goes with wrong or right answers.

An attempt is one eval row. Its records (one per turn) are summed first:
share(layer, expert) = mass(layer, expert) / mass(layer). An expert absent from that layer's recorded top list
contributes 0 (the list is the top-K only; coverage is the fraction of layer mass those entries hold).

lift = (mean share on incorrect attempts + eps) / (mean share on correct attempts + eps), eps = 1e-6.
eps is in share units, far below a selected expert (typically ~0.01 to ~0.3 of the layer), so a zero mean
does not send the ratio to infinity and a real share is barely moved.

The 95% interval is a seeded bootstrap (default 1000 replicates, seed 0). The resample unit is the task:
draw the domain's tasks with replacement and keep every attempt of a drawn task, so draws of one task are
not treated as independent. A replicate with no correct or no incorrect attempt is dropped. The interval is
the type-1 quantile of the rest: order statistics at ceil(0.025*R) and ceil(0.975*R), 1-based.

Within-task estimate: over tasks that have both outcomes, the mean of (mean incorrect share - mean correct
share). Each such task counts once; its interval resamples those tasks. support is how many attempts the
expert appears in (mass or count > 0). Below --min-attempts (default 10), or with no interval, no verdict.
Suspect: lower bound > 1 (rank by that bound, then within-task, both descending). Keeper: upper bound < 1
(rank by that bound ascending, then within-task ascending). --mask-out writes suspects as layer:expert.

Association only. Task difficulty, the arm, and mass cut off by the top-K list all confound it.
"""
import argparse, json, math, os, random, sys

sys.dont_write_bytecode = True
EPS = 1e-6


def ci95(xs):
    """Type-1 95% quantile: smallest order statistic whose ECDF reaches 0.025 and 0.975."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    def at(p):
        return s[min(max(math.ceil(p * n) - 1, 0), n - 1)]
    return [at(0.025), at(0.975)]


def _samples(rng, n, boots):
    if n <= 0 or boots <= 0:
        return []
    return [[rng.randrange(n) for _ in range(n)] for _ in range(boots)]


def _as_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    return int(v) if isinstance(v, float) and v.is_integer() else None


def _as_float(v):
    return None if isinstance(v, bool) or not isinstance(v, (int, float)) else float(v)


def _parse_record(obj):
    if not isinstance(obj, dict):
        return None
    rid, layers = obj.get("id"), obj.get("layers")
    if not isinstance(rid, str) or not rid or not isinstance(layers, list):
        return None
    parsed = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        l, mass = _as_int(layer.get("l")), _as_float(layer.get("mass"))
        if l is None or mass is None:
            continue
        top = []
        for item in layer.get("top") or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            e, em = _as_int(item[0]), _as_float(item[1])
            if e is None or em is None:
                continue
            ec = _as_int(item[2]) if len(item) > 2 else 0
            top.append((e, em, 0 if ec is None else ec))
        parsed.append((l, mass, top))
    return {"id": rid, "layers": parsed}


def record_paths(path):
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.startswith("route-") and n.endswith(".jsonl"))
        return [os.path.join(path, n) for n in names]
    return [path]


def load_records(paths):
    """id -> parsed record. The first line with an id wins; later copies count as duplicate_ids."""
    store, seen = {}, set()
    cov = {"truncated_lines": 0, "duplicate_ids": 0, "records_rejected": 0, "records_loaded": 0}
    for path in paths:
        for file in record_paths(path):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        cov["truncated_lines"] += 1
                        continue
                    rec = _parse_record(obj)
                    if rec is None:
                        cov["records_rejected"] += 1
                        continue
                    if rec["id"] in seen:
                        cov["duplicate_ids"] += 1
                        continue
                    seen.add(rec["id"])
                    store[rec["id"]] = rec
                    cov["records_loaded"] += 1
    return store, cov


def _ids_of(row):
    """route_records (list, nulls dropped) wins over a single route_record."""
    if isinstance(row.get("route_records"), list):
        return [x for x in row["route_records"] if isinstance(x, str) and x]
    one = row.get("route_record")
    return [one] if isinstance(one, str) and one else []


def _correct(row):
    score = row.get("score")
    if not isinstance(score, dict):
        return None
    c = score.get("correct")
    return c if isinstance(c, bool) else None


def load_rows(paths):
    rows, bad = [], 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad += 1
    return rows, bad


def _observe(recs):
    """Sum an attempt's records. Returns {(layer, expert): (share, mass, count)} and per-layer top-K fractions."""
    layer_mass, acc, fracs = {}, {}, []
    for rec in recs:
        for l, mass, top in rec["layers"]:
            layer_mass[l] = layer_mass.get(l, 0.0) + mass
            listed = 0.0
            for e, em, ec in top:
                listed += em
                slot = acc.setdefault((l, e), [0.0, 0])
                slot[0] += em
                slot[1] += ec
            if mass > 0:
                fracs.append(listed / mass)
    out = {}
    for key, (em, ec) in acc.items():
        if em == 0 and ec == 0:
            continue
        tot = layer_mass.get(key[0], 0.0)
        out[key] = (em / tot if tot > 0 else 0.0, em, ec)
    return out, fracs


def _add(ag, correct, shares):
    ag["n_c" if correct else "n_i"] += 1
    for key, (share, mass, count) in shares.items():
        slot = ag["ex"].get(key)
        if slot is None:
            slot = [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0]  # share, mass, count, appear — correct then incorrect
            ag["ex"][key] = slot
        side = 0 if correct else 1
        slot[side] += share
        slot[2 + side] += mass
        slot[4 + side] += count
        slot[6 + side] += 1


def _lift_reps(n_c, n_i, sum_c, sum_i, samples, eps):
    reps = []
    for samp in samples:
        nc = ni = 0
        sc = si = 0.0
        for t in samp:
            nc += n_c[t]; ni += n_i[t]; sc += sum_c[t]; si += sum_i[t]
        if nc and ni:
            reps.append((si / ni + eps) / (sc / nc + eps))
    return reps


def _support(tasks, task_ids, key):
    return sum(tasks[t]["ex"][key][6] + tasks[t]["ex"][key][7] for t in task_ids if key in tasks[t]["ex"])


def _tie(cell):
    return 0.0 if cell["within_task"] is None else cell["within_task"]


def _brief(cell):
    return {"layer": cell["layer"], "expert": cell["expert"], "lift": cell["lift"], "ci": cell["ci"],
            "within_task": cell["within_task"], "support": cell["support"]}


def _cell(tasks, task_ids, mixed, key, n_c, n_i, N_c, N_i, attempts, min_attempts, eps, samples, samples_w):
    sum_c, sum_i = [], []
    mass_c = mass_i = 0.0
    count_c = count_i = app_c = app_i = 0
    for t in task_ids:
        slot = tasks[t]["ex"].get(key)
        if slot is None:
            sum_c.append(0.0); sum_i.append(0.0); continue
        sum_c.append(slot[0]); sum_i.append(slot[1])
        mass_c += slot[2]; mass_i += slot[3]
        count_c += slot[4]; count_i += slot[5]
        app_c += slot[6]; app_i += slot[7]
    support = app_c + app_i
    mean_c = sum(sum_c) / N_c if N_c else None
    mean_i = sum(sum_i) / N_i if N_i else None
    lift = (mean_i + eps) / (mean_c + eps) if mean_c is not None and mean_i is not None else None
    deltas = [(sum_i[i] / n_i[i] - sum_c[i] / n_c[i]) for i in mixed]
    within = sum(deltas) / len(deltas) if deltas else None
    ci = within_ci = verdict = None
    if support >= min_attempts and lift is not None and samples:
        ci = ci95(_lift_reps(n_c, n_i, sum_c, sum_i, samples, eps))
        within_ci = ci95([sum(deltas[t] for t in samp) / len(samp) for samp in samples_w]) if deltas and samples_w else None
        if ci is not None and ci[0] > 1:
            verdict = "suspect"
        elif ci is not None and ci[1] < 1:
            verdict = "keeper"
    return {"layer": key[0], "expert": key[1], "attempts": attempts, "appear_correct": app_c, "appear_incorrect": app_i,
            "mean_share_correct": mean_c, "mean_share_incorrect": mean_i, "mass_correct": mass_c, "mass_incorrect": mass_i,
            "count_correct": count_c, "count_incorrect": count_i, "lift": lift, "ci": ci, "within_task": within,
            "within_task_ci": within_ci, "within_tasks": len(deltas), "support": support, "verdict": verdict}


def _domain_report(tasks, fracs, min_attempts, boot, seed, top, eps, domain):
    task_ids = sorted(tasks)
    n_c = [tasks[t]["n_c"] for t in task_ids]
    n_i = [tasks[t]["n_i"] for t in task_ids]
    N_c, N_i = sum(n_c), sum(n_i)
    mixed = [i for i, t in enumerate(task_ids) if tasks[t]["n_c"] and tasks[t]["n_i"]]
    keys = sorted({k for t in task_ids for k in tasks[t]["ex"]})
    samples = samples_w = None
    if N_c and N_i and boot > 0 and any(_support(tasks, task_ids, k) >= min_attempts for k in keys):
        rng = random.Random("%d\0%s" % (seed, domain))
        samples = _samples(rng, len(task_ids), boot)
        samples_w = _samples(rng, len(mixed), boot)
    cells = [_cell(tasks, task_ids, mixed, key, n_c, n_i, N_c, N_i, N_c + N_i, min_attempts, eps, samples, samples_w) for key in keys]
    suspects = sorted((c for c in cells if c["verdict"] == "suspect"), key=lambda c: (-c["ci"][0], -_tie(c), c["layer"], c["expert"]))
    keepers = sorted((c for c in cells if c["verdict"] == "keeper"), key=lambda c: (c["ci"][1], _tie(c), c["layer"], c["expert"]))
    return {"attempts": N_c + N_i, "attempts_correct": N_c, "attempts_incorrect": N_i, "tasks": len(task_ids), "cells": cells,
            "suspects": [_brief(c) for c in suspects[:top]], "keepers": [_brief(c) for c in keepers[:top]],
            "coverage_topk_mass_fraction": (sum(fracs) / len(fracs) if fracs else None)}


def join(row_paths, record_paths_in, domain_key="suite", min_attempts=10, boot=1000, seed=0, top=20, eps=EPS, cluster="task"):
    store, cov = load_records(record_paths_in)
    rows, bad = load_rows(row_paths)
    skipped = {"bad_row": bad, "no_score": 0, "no_ids": 0, "no_record": 0}
    referenced, missing, domains, fracs, used, seq = set(), 0, {}, {}, 0, 0
    for row in rows:
        correct, ids = _correct(row), _ids_of(row)
        for rid in ids:
            referenced.add(rid)
            missing += rid not in store
        if correct is None:
            skipped["no_score"] += 1
            continue
        if not ids:
            skipped["no_ids"] += 1
            continue
        got = [store[r] for r in ids if r in store]
        if not got:
            skipped["no_record"] += 1
            continue
        shares, got_fracs = _observe(got)
        domain = row.get(domain_key)
        domain = "" if domain is None else str(domain)
        task = "%08d" % seq if cluster == "row" else str("" if row.get("task") is None else row.get("task"))
        seq += 1
        ag = domains.setdefault(domain, {}).setdefault(task, {"n_c": 0, "n_i": 0, "ex": {}})
        _add(ag, correct, shares)
        fracs.setdefault(domain, []).extend(got_fracs)
        used += 1
    cov.update({"skipped_rows": skipped, "records_missing": missing, "rows_read": len(rows) + bad, "rows_used": used,
                "records_unmatched": sum(1 for rid in store if rid not in referenced)})
    built = {d: _domain_report(domains[d], fracs.get(d, []), min_attempts, boot, seed, top, eps, d) for d in sorted(domains)}
    cov["domains"] = {d: {"mean_topk_mass_fraction": built[d]["coverage_topk_mass_fraction"], "attempts": built[d]["attempts"]} for d in built}
    return {"v": 1, "method": _method(eps, boot, seed, min_attempts, cluster), "domains": built, "coverage": cov}


def _method(eps, boot, seed, min_attempts, cluster):
    return {"share": "mass(layer, expert) / mass(layer) after summing the attempt's records; experts absent from the top list contribute 0",
            "lift": "(mean_share_incorrect + eps) / (mean_share_correct + eps)", "eps": eps,
            "ci": "95% type-1 quantile of a bootstrap that resamples tasks with replacement and keeps every attempt of a drawn task; replicates missing either outcome are dropped",
            "within_task": "mean over tasks with both outcomes of (mean incorrect share - mean correct share); its CI resamples those tasks",
            "boot": boot, "seed": seed, "min_attempts": min_attempts, "cluster": cluster,
            "support": "attempts in which the expert appears (mass or count > 0); below min_attempts a cell has no verdict",
            "suspect": "lift CI lower bound > 1, ranked by that bound then by within_task, both descending",
            "keeper": "lift CI upper bound < 1, ranked by that bound ascending, then by within_task ascending"}


def write_mask(report, path):
    lines, seen = [], set()
    for domain in sorted(report["domains"]):
        for s in report["domains"][domain]["suspects"]:
            item = "%d:%d" % (s["layer"], s["expert"])
            if item not in seen:
                seen.add(item)
                lines.append(item)
    with open(path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def print_table(report):
    print("%-16s %-8s %5s %7s %8s %8s %8s %8s %7s" % ("domain", "kind", "layer", "expert", "lift", "ci_lo", "ci_hi", "within", "support"))
    any_row = False
    for domain in sorted(report["domains"]):
        d = report["domains"][domain]
        for kind, rows in (("suspect", d["suspects"]), ("keeper", d["keepers"])):
            for s in rows:
                any_row = True
                w = s["within_task"]
                print("%-16s %-8s %5d %7d %8.4f %8.4f %8.4f %8s %7d" % (
                    str(domain)[:16], kind, s["layer"], s["expert"], s["lift"], s["ci"][0], s["ci"][1],
                    "n/a" if w is None else "%.4f" % w, s["support"]))
    if not any_row:
        print("(no suspect or keeper)")
    c, sk = report["coverage"], report["coverage"]["skipped_rows"]
    print("coverage rows_used=%d skipped=%s unmatched=%d missing=%d truncated=%d duplicate_ids=%d" % (
        c["rows_used"], ",".join("%s=%d" % (k, sk[k]) for k in sorted(sk)), c["records_unmatched"], c["records_missing"],
        c["truncated_lines"], c["duplicate_ids"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="route_join", description="Join eval verdicts to route records and rank experts by lift.")
    ap.add_argument("--rows", action="append", required=True, help="eval rows JSONL (repeatable)")
    ap.add_argument("--records", action="append", required=True, help="route-record JSONL file or directory of route-*.jsonl")
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--mask-out", help="write suspect experts as layer:expert lines")
    ap.add_argument("--domain-key", default="suite", help="row field that names the domain (default: suite)")
    ap.add_argument("--min-attempts", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000, help="task-cluster bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--eps", type=float, default=EPS, help="added to both lift means (default 1e-6)")
    a = ap.parse_args(argv)
    try:
        report = join(a.rows, a.records, domain_key=a.domain_key, min_attempts=a.min_attempts, boot=a.boot, seed=a.seed, top=a.top, eps=a.eps)
    except FileNotFoundError as e:
        print("route_join: %s" % e, file=sys.stderr)
        return 2
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if a.mask_out:
        write_mask(report, a.mask_out)
    print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
