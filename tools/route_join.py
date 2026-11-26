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

Multiple comparisons. ~15,000 (layer, expert) cells are tested at once, so the CI verdict above is not what
feeds a mask. Each cell with support >= min_attempts gets a two-sided permutation p-value on |log lift| (the
lift statistic above). The null shuffles correct/incorrect labels within (domain, arm) strata and keeps each
task's structure: a task's attempts stay together and the per-task correct counts are only moved between
tasks that have the same number of attempts in the stratum (then which attempts carry them is shuffled inside
the task). So a task with all-incorrect draws stays all-incorrect, and per-stratum and per-task label totals
are exactly preserved. p = (1 + #{perm stat >= observed}) / (perms + 1).
Primary q-value (--fdr perm, default): permutation FDR. For every permutation each cell's statistic gets a p
by its rank among that cell's permuted values (counted with itself, over perms + 1); V(t) = (1 + total over permutations of the number of
cells with p <= t) / perms, R(t) = observed cells with p <= t (V counts one extra false cell, so q is never 0), q = running minimum over t >= p of min(1, V(t)/R(t)).
It keeps the correlation between cells (shares of one layer sum to 1) in the null; pi0 is not estimated, so it
is conservative. Second q (--fdr bh): Benjamini-Hochberg on the per-cell permutation p-values. Both are
reported beside the lift; the family is every tested cell of every domain. With P permutations the smallest p
is 1/(P+1), so a q below ~cells/((P+1)*hits) cannot be reached: raise --perms for fewer hits.
Suspect: lift > 1 with q < alpha (--alpha, default 0.05); keeper: lift < 1 with q < alpha; rank by q, p, then
lift. --mask-out writes only those suspects as layer:expert, after a '#' header with method, alpha, perms, seed.
The bootstrap CI and its 'verdict' stay as descriptive fields and no longer gate anything.

Association only. Task difficulty, the arm, and mass cut off by the top-K list all confound it.
"""
import argparse, bisect, json, math, os, random, sys

try:
    import numpy as np
except ImportError:  # pure-python fallback: same answers, slow at real sizes
    np = None

sys.dont_write_bytecode = True
EPS = 1e-6
ROUND = 10  # permutation statistics are rounded so float noise cannot split exact ties


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
            "within_task": cell["within_task"], "support": cell["support"], "p": cell["p_perm"],
            "q": cell["q_perm"] if cell["_primary"] == "perm" else cell["q_bh"], "q_perm": cell["q_perm"], "q_bh": cell["q_bh"]}


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
            "within_task_ci": within_ci, "within_tasks": len(deltas), "support": support, "verdict": verdict,
            "p_perm": None, "q_perm": None, "q_bh": None, "fdr_verdict": None}


def _log_stat(si, tot, n_i, n_c, eps):
    """|log lift| for cells whose incorrect-attempt share sums are si (tot is the all-attempt sum)."""
    if np is not None:
        return np.round(np.abs(np.log((si / n_i + eps) / ((tot - si) / n_c + eps))), ROUND)
    return [round(abs(math.log((a / n_i + eps) / ((t - a) / n_c + eps))), ROUND) for a, t in zip(si, tot)]


class _PermEngine:
    """Label permutations of one domain. attempts: [(arm, task, correct, shares)]; cells: tested (layer, expert) keys."""

    def __init__(self, attempts, cells, eps):
        self.n = len(attempts)
        self.eps = eps
        self.n_c = sum(1 for a in attempts if a[2])
        self.n_i = self.n - self.n_c
        index = {k: i for i, k in enumerate(cells)}
        per_cell = [[] for _ in cells]
        tot = [0.0] * len(cells)
        for ai, (_, _, _, shares) in enumerate(attempts):
            for key, (share, _, _) in shares.items():
                ci = index.get(key)
                if ci is not None:
                    per_cell[ci].append((ai, share))
                    tot[ci] += share
        self.per_cell = per_cell
        if np is not None:
            self.tot = np.array(tot)
            self.att = np.array([a for lst in per_cell for a, _ in lst], dtype=np.int64)
            self.cell = np.array([c for c, lst in enumerate(per_cell) for _ in lst], dtype=np.int64)
            self.share = np.array([s for lst in per_cell for _, s in lst])
        else:
            self.tot = tot
        blocks = {}
        for ai, (arm, task, _, _) in enumerate(attempts):
            blocks.setdefault((arm, task), []).append(ai)
        groups = {}
        for (arm, _), idx in sorted(blocks.items()):
            groups.setdefault((arm, len(idx)), []).append(idx)
        self.groups = [g for _, g in sorted(groups.items())]
        self.observed_incorrect = [not a[2] for a in attempts]

    def labels(self, rng):
        """Incorrect flags after moving per-task correct counts among equal-size tasks of a stratum."""
        inc = [True] * self.n
        for blocks in self.groups:
            counts = [sum(1 for a in b if not self.observed_incorrect[a]) for b in blocks]  # correct count of each task
            rng.shuffle(counts)
            for b, k in zip(blocks, counts):
                for a in rng.sample(b, k):
                    inc[a] = False
        return inc

    def stat(self, inc):
        if np is not None:
            w = np.bincount(self.cell, weights=self.share * np.array(inc, dtype=float)[self.att], minlength=len(self.per_cell))
        else:
            w = [sum(s for a, s in lst if inc[a]) for lst in self.per_cell]
        return _log_stat(w, self.tot, self.n_i, self.n_c, self.eps)

    def run(self, perms, rng):
        """Rows: observed first, then perms permuted statistics."""
        rows = [self.stat(self.observed_incorrect)]
        for _ in range(perms):
            rows.append(self.stat(self.labels(rng)))
        return rows


def _p_matrix(obs, rows):
    """p_obs[c] = (1 + #{perm >= obs}) / (B + 1); p_perm[b][c] = #{perm >= perm b, itself included} / (B + 1), B = len(rows).
    A permuted statistic is ranked inside the permutation pool only, so the observed value cannot sit above the null."""
    n = len(rows)
    if np is not None:
        mat = np.array(rows)
        p_obs = (1 + (mat >= np.array(obs)).sum(axis=0)) / (n + 1)
        order = np.argsort(mat, axis=0, kind="stable")
        srt = np.take_along_axis(mat, order, axis=0)
        idx = np.arange(n)[:, None] * np.ones((1, mat.shape[1]), dtype=np.int64)
        new = np.ones(mat.shape, dtype=bool)
        new[1:] = srt[1:] != srt[:-1]
        start = np.maximum.accumulate(np.where(new, idx, 0), axis=0)
        out = np.empty(mat.shape)
        np.put_along_axis(out, order, (n - start) / (n + 1), axis=0)
        return p_obs, out
    cols = list(zip(*rows)) if rows and rows[0] else []
    p_obs, pcols = [], []
    for o, col in zip(obs, cols):
        srt = sorted(col)
        p_obs.append((1 + n - bisect.bisect_left(srt, o)) / (n + 1))
        pcols.append([(n - bisect.bisect_left(srt, v)) / (n + 1) for v in col])
    return p_obs, [list(r) for r in zip(*pcols)]


def bh_qvalues(ps):
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1), in the order given."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q, run = [None] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        q[i] = run
    return q


def perm_fdr_qvalues(p_obs, p_perm_flat, perms):
    """Permutation FDR. p_obs: observed p per cell; p_perm_flat: every permuted p of every cell, all permutations."""
    srt = sorted(p_perm_flat)
    obs_sorted = sorted(p_obs)
    fdr = []
    for p in p_obs:
        v = (1 + bisect.bisect_right(srt, p)) / perms
        r = bisect.bisect_right(obs_sorted, p)
        fdr.append(min(1.0, v / r))
    order = sorted(range(len(p_obs)), key=lambda i: p_obs[i])
    q, run = [None] * len(p_obs), 1.0
    for i in reversed(order):
        run = min(run, fdr[i])
        q[i] = run
    return q


def _fdr_np_qvalues(p_obs, perm_blocks, perms):
    """perm_fdr_qvalues with numpy sorting for the 15M-value case."""
    srt = np.sort(np.concatenate([b.ravel() for b in perm_blocks]))
    p = np.array(p_obs)
    v = (1 + np.searchsorted(srt, p, side="right")) / perms
    r = np.searchsorted(np.sort(p), p, side="right")
    fdr = np.minimum(1.0, v / r)
    order = np.argsort(p, kind="stable")
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate(fdr[order][::-1])[::-1]
    return [float(x) for x in q]


def _domain_report(tasks, fracs, min_attempts, boot, seed, eps, domain, attempts, perms):
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
    tested = [c for c in cells if c["support"] >= min_attempts and c["lift"] is not None]
    perm = None
    if tested and perms > 0:
        engine = _PermEngine(attempts, [(c["layer"], c["expert"]) for c in tested], eps)
        stats = engine.run(perms, random.Random("%d\0perm\0%s" % (seed, domain)))
        p_obs, p_perm = _p_matrix(stats[0], stats[1:])
        perm = {"cells": tested, "p_obs": [float(x) for x in p_obs], "p_perm": p_perm}
    return {"attempts": N_c + N_i, "attempts_correct": N_c, "attempts_incorrect": N_i, "tasks": len(task_ids), "cells": cells,
            "tested_cells": len(tested), "coverage_topk_mass_fraction": (sum(fracs) / len(fracs) if fracs else None)}, perm


def _apply_fdr(built, perms_by_domain, perms, alpha, fdr, top):
    """Pool every tested cell of every domain, set p/q/fdr_verdict on the cells, fill suspects and keepers."""
    pooled = [(d, p) for d, p in perms_by_domain.items() if p]
    cells = [c for _, p in pooled for c in p["cells"]]
    p_obs = [x for _, p in pooled for x in p["p_obs"]]
    if cells:
        if np is not None:
            q_perm = _fdr_np_qvalues(p_obs, [np.array(p["p_perm"]) for _, p in pooled], perms)
        else:
            q_perm = perm_fdr_qvalues(p_obs, [x for _, p in pooled for row in p["p_perm"] for x in row], perms)
        q_bh = bh_qvalues(p_obs)
        for c, p0, qp, qb in zip(cells, p_obs, q_perm, q_bh):
            c["p_perm"], c["q_perm"], c["q_bh"] = p0, qp, qb
            q = qp if fdr == "perm" else qb
            if q < alpha:
                c["fdr_verdict"] = "suspect" if c["lift"] > 1 else "keeper" if c["lift"] < 1 else None
    for d in built.values():
        for c in d["cells"]:
            c["_primary"] = fdr
        sus = sorted((c for c in d["cells"] if c["fdr_verdict"] == "suspect"),
                     key=lambda c: (_brief(c)["q"], c["p_perm"], -c["lift"], -_tie(c), c["layer"], c["expert"]))
        kee = sorted((c for c in d["cells"] if c["fdr_verdict"] == "keeper"),
                     key=lambda c: (_brief(c)["q"], c["p_perm"], c["lift"], _tie(c), c["layer"], c["expert"]))
        d["suspects"], d["keepers"] = [_brief(c) for c in sus[:top]], [_brief(c) for c in kee[:top]]
        d["mask"] = [_brief(c) for c in sus]  # every suspect under the threshold, not cut to top
        d["suspects_total"], d["keepers_total"] = len(sus), len(kee)
        for c in d["cells"]:
            del c["_primary"]


def join(row_paths, record_paths_in, domain_key="suite", min_attempts=10, boot=1000, seed=0, top=20, eps=EPS, cluster="task",
         perms=1000, alpha=0.05, fdr="perm", arm_key="arm"):
    if fdr not in ("perm", "bh"):
        raise ValueError("fdr must be perm or bh")
    store, cov = load_records(record_paths_in)
    rows, bad = load_rows(row_paths)
    skipped = {"bad_row": bad, "no_score": 0, "no_ids": 0, "no_record": 0}
    referenced, missing, domains, fracs, used, seq, attempts = set(), 0, {}, {}, 0, 0, {}
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
        arm = row.get(arm_key)
        attempts.setdefault(domain, []).append(("" if arm is None else str(arm), task, correct, shares))
        fracs.setdefault(domain, []).extend(got_fracs)
        used += 1
    cov.update({"skipped_rows": skipped, "records_missing": missing, "rows_read": len(rows) + bad, "rows_used": used,
                "records_unmatched": sum(1 for rid in store if rid not in referenced)})
    built, perm_data = {}, {}
    for d in sorted(domains):
        built[d], perm_data[d] = _domain_report(domains[d], fracs.get(d, []), min_attempts, boot, seed, eps, d, attempts[d], perms)
    _apply_fdr(built, perm_data, perms, alpha, fdr, top)
    cov["domains"] = {d: {"mean_topk_mass_fraction": built[d]["coverage_topk_mass_fraction"], "attempts": built[d]["attempts"]} for d in built}
    return {"v": 2, "method": _method(eps, boot, seed, min_attempts, cluster, perms, alpha, fdr, arm_key), "domains": built, "coverage": cov}


def _method(eps, boot, seed, min_attempts, cluster, perms, alpha, fdr, arm_key):
    return {"share": "mass(layer, expert) / mass(layer) after summing the attempt's records; experts absent from the top list contribute 0",
            "lift": "(mean_share_incorrect + eps) / (mean_share_correct + eps)", "eps": eps,
            "ci": "95% type-1 quantile of a bootstrap that resamples tasks with replacement and keeps every attempt of a drawn task; replicates missing either outcome are dropped",
            "within_task": "mean over tasks with both outcomes of (mean incorrect share - mean correct share); its CI resamples those tasks",
            "boot": boot, "seed": seed, "min_attempts": min_attempts, "cluster": cluster,
            "support": "attempts in which the expert appears (mass or count > 0); below min_attempts a cell has no verdict",
            "ci_verdict": "descriptive: lift CI lower bound > 1 (suspect) or upper bound < 1 (keeper); does not gate suspects, keepers or the mask",
            "fdr": {"primary": fdr, "alpha": alpha, "perms": perms, "seed": seed, "strata": ["domain", arm_key],
                    "statistic": "|log lift| per cell, two-sided; cells with support >= min_attempts are tested",
                    "null": "labels shuffled within (domain, arm) strata; per-task correct counts move only between tasks of equal attempt count, then which attempts carry them is shuffled inside the task",
                    "p_perm": "(1 + #{permuted stat >= observed}) / (perms + 1), per cell",
                    "q_perm": "permutation FDR: V(t) = (1 + sum over permutations of cells with p <= t (each permuted statistic ranked among its cell's permuted values, #>= / (perms + 1)), R(t) = observed cells with p <= t, q = running min of min(1, V/R) over t >= p; pi0 not estimated; family = all tested cells of all domains",
                    "q_bh": "Benjamini-Hochberg on p_perm over the same family"},
            "suspect": "lift > 1 and q < alpha (q of the primary method), ranked by q, p_perm, lift descending",
            "keeper": "lift < 1 and q < alpha, ranked by q, p_perm, lift ascending"}


def write_mask(report, path):
    """Suspects under the FDR threshold only (not the --top cut). A '#' header records the method."""
    f_ = report["method"]["fdr"]
    lines, seen = [], set()
    for domain in sorted(report["domains"]):
        for s in report["domains"][domain].get("mask", []):
            item = "%d:%d" % (s["layer"], s["expert"])
            if item not in seen:
                seen.add(item)
                lines.append(item)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# route_join mask: suspects with q < %s, method=%s-fdr, permutations=%d, seed=%d, min_attempts=%d, strata=domain+%s\n" % (
            f_["alpha"], f_["primary"], f_["perms"], f_["seed"], report["method"]["min_attempts"], f_["strata"][1]))
        if lines:
            f.write("\n".join(lines) + "\n")


def print_table(report):
    print("%-16s %-8s %5s %7s %8s %8s %8s %8s %8s %8s %7s" % ("domain", "kind", "layer", "expert", "lift", "q_perm", "q_bh", "ci_lo", "ci_hi", "within", "support"))
    any_row = False
    for domain in sorted(report["domains"]):
        d = report["domains"][domain]
        for kind, rows in (("suspect", d["suspects"]), ("keeper", d["keepers"])):
            for s in rows:
                any_row = True
                w = s["within_task"]
                ci = s["ci"] or [float("nan")] * 2
                print("%-16s %-8s %5d %7d %8.4f %8.4g %8.4g %8.4f %8.4f %8s %7d" % (
                    str(domain)[:16], kind, s["layer"], s["expert"], s["lift"], s["q_perm"], s["q_bh"], ci[0], ci[1],
                    "n/a" if w is None else "%.4f" % w, s["support"]))
    if not any_row:
        print("(no cell under the FDR threshold)")
    c, sk = report["coverage"], report["coverage"]["skipped_rows"]
    print("coverage rows_used=%d skipped=%s unmatched=%d missing=%d truncated=%d duplicate_ids=%d" % (
        c["rows_used"], ",".join("%s=%d" % (k, sk[k]) for k in sorted(sk)), c["records_unmatched"], c["records_missing"],
        c["truncated_lines"], c["duplicate_ids"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="route_join", description="Join eval verdicts to route records and rank experts by lift.")
    ap.add_argument("--rows", action="append", required=True, help="eval rows JSONL (repeatable)")
    ap.add_argument("--records", action="append", required=True, help="route-record JSONL file or directory of route-*.jsonl")
    ap.add_argument("--out", required=True, help="JSON report path")
    ap.add_argument("--mask-out", help="write suspects with q < alpha as layer:expert lines under a # header")
    ap.add_argument("--domain-key", default="suite", help="row field that names the domain (default: suite)")
    ap.add_argument("--min-attempts", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000, help="task-cluster bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000, help="label permutations for the FDR null (0 disables the FDR, so no verdicts)")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR threshold: suspects, keepers and the mask need q < alpha")
    ap.add_argument("--fdr", choices=("perm", "bh"), default="perm", help="which q gates the lists and the mask (both are reported)")
    ap.add_argument("--arm-key", default="arm", help="row field that names the arm, a permutation stratum with the domain (default: arm)")
    ap.add_argument("--eps", type=float, default=EPS, help="added to both lift means (default 1e-6)")
    a = ap.parse_args(argv)
    try:
        report = join(a.rows, a.records, domain_key=a.domain_key, min_attempts=a.min_attempts, boot=a.boot, seed=a.seed, top=a.top, eps=a.eps,
                      perms=a.perms, alpha=a.alpha, fdr=a.fdr, arm_key=a.arm_key)
    except (FileNotFoundError, ValueError) as e:
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
