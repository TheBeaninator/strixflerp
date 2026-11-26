#!/usr/bin/env python3
"""tools/ablate.py: the expert-mask ablation driver for DeepSeek-V4.1-Flash (pod card 5bb471d3b9).

The engine side is the StrixFold fork's MASK_EXPERTS=<file> (serve.conf; TF_DS_MASK_EXPERTS on every rank): the
listed routed experts get -inf routing bias at load, so they never enter top-k and the top-k weights renormalise over
the rest, as a checkpoint with those experts deleted would route. This tool measures what a mask costs, given one
endpoint without the mask and one with it. It never starts, stops or configures a server.

  kl    a frozen calibration prompt set (JSONL rows {"id", "domain", "messages", optional "body"}) sent greedily
        (temperature 0, one seed) to both endpoints with top-k logprobs; per prompt: the KL divergence
        KL(unmasked || masked) of the next-token distribution at the first generated position (the prompt's own
        next-token distribution, the same context on both), the same KL pooled over every position whose context is
        still identical (up to and including the first position where the greedy tokens part), and token agreement
        (the share of the N positions whose greedy tokens are equal, and the agreeing prefix). Summary: mean and p95 KL,
        mean agreement, overall and per domain. KL is over the shared top-K support (the tokens both endpoints list at
        the position, from the server's top_logprobs) plus one remainder bucket holding each side's mass outside it:
        the exact KL of both distributions coarsened the same way, so a lower bound on the full-vocabulary KL (pod card
        73bab11681). An endpoint that refuses logprobs (dsv41 builds before pod card 73bab11681: fields.zig
        probabilityOptions) is asked again without them: KL is then null with kl_status naming why, and agreement uses
        the endpoint's /tokenize of each reply instead of the reply's tokens.
  gate  an FQ-1 gate run twice through the eval harness's run.py (--eval-repo): the masked run with
        --engine ENGINE=MASKED_URL into OUT/masked (OUT/masked/ablation.json marks the engine masked, with the mask's
        path, file sha256 and canonical id), and the unmasked run into OUT/base (or --base-run, an earlier run of the same
        gate and arms). Arm names are the gate's own, so both runs draw the same seeds (gatefile.seed hashes task, arm
        and draw) and their rows pair by (task, arm, draw). OUT/deltas.json and deltas.md: per arm and suite, the paired
        accuracy unmasked and masked, the delta with a family-cluster bootstrap 95 % interval, and the discordant
        counts. --analyze also runs analyze.py on each run (the validity report per run).

The mask id is the engine's: SHA-256 of the canonical list ("layer:e1,e2,...\n" per masked layer, both ascending), its
first 16 hex characters, as the plan text's " mask=<id>" shows it (mask.zig canonicalHash). Stdlib only.
"""
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

LAYERS, EXPERTS, TOP_K = 40, 384, 6   # DeepSeek-V4.1-Flash target layers, routed experts, experts a token
SCORED = ("ok", "loop")               # analyze.py's scored statuses ("loop" is scored wrong)


# -- the mask file (mask.zig's rules) ---------------------------------------------------------------------------

class MaskError(ValueError):
    pass


def _num(s):
    s = s.strip(" \t")
    if not s or len(s) > 9 or not all("0" <= c <= "9" for c in s):
        return None
    return int(s)


def parse_mask(text, layers=LAYERS, experts=EXPERTS, top_k=TOP_K):
    """{layer: set(experts)} from a mask file's text, refused (MaskError naming the line) as the engine refuses it: a
    line that is not 'layer:e[,e...]', a layer past the target layers (an MTP block), an expert past the routed ones
    (the shared expert's id) or a layer left with fewer than top_k experts. '#' starts a comment; blank lines skip."""
    mask = {}
    for no, raw in enumerate(text.split("\n"), 1):
        line = raw.split("#", 1)[0].strip(" \t\r")
        if not line:
            continue
        if ":" not in line:
            raise MaskError("line %d: not 'layer:expert[,expert...]': %r" % (no, raw))
        head, tail = line.split(":", 1)
        layer = _num(head)
        if layer is None:
            raise MaskError("line %d: bad layer: %r" % (no, raw))
        if layer >= layers:
            raise MaskError("line %d: layer %d is past the %d target layers" % (no, layer, layers))
        for part in tail.split(","):
            e = _num(part)
            if e is None:
                raise MaskError("line %d: bad expert: %r" % (no, raw))
            if e >= experts:
                raise MaskError("line %d: expert %d is past the %d routed experts" % (no, e, experts))
            mask.setdefault(layer, set()).add(e)
    for layer, es in mask.items():
        if experts - len(es) < top_k:
            raise MaskError("layer %d keeps %d experts, fewer than top_k = %d" % (layer, experts - len(es), top_k))
    return mask


def canonical(mask):
    return "".join("%d:%s\n" % (l, ",".join(str(e) for e in sorted(mask[l]))) for l in sorted(mask) if mask[l])


def mask_id(mask):
    """The engine's name for the mask (16 hex), or "" for a mask that masks nothing (the stock plan text)."""
    c = canonical(mask)
    return hashlib.sha256(c.encode()).hexdigest()[:16] if c else ""


def load_mask(path):
    raw = read_bytes(path)
    mask = parse_mask(raw.decode("utf-8"))
    return mask, {"path": os.path.abspath(path), "file_sha256": hashlib.sha256(raw).hexdigest(), "id": mask_id(mask),
                  "experts_masked": sum(len(v) for v in mask.values()), "layers": sorted(mask)}


# -- HTTP -------------------------------------------------------------------------------------------------------

def post_json(url, body, timeout):
    """(HTTP status, parsed JSON or text); a refused connection is status 0."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        data = e.read().decode("utf-8", "replace")
        status = e.code
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)
    try:
        return status, json.loads(data)
    except ValueError:
        return status, data


def refuses_logprobs(status, reply):
    return status == 400 and "logprobs" in json.dumps(reply).lower()


def chat_body(prompt, model, max_tokens, top_logprobs, extra=None):
    b = {"model": model, "messages": prompt["messages"], "max_tokens": max_tokens, "temperature": 0, "seed": 0,
         "stream": False}
    if top_logprobs:
        b["logprobs"] = True
        b["top_logprobs"] = top_logprobs
    b.update(extra or {})
    b.update(prompt.get("body") or {})
    return b


def token_key(entry):
    """A logprobs entry's identity: its bytes (the fork server gives every partial UTF-8 token the same text, U+FFFD,
    and different bytes), else its text's UTF-8."""
    b = entry.get("bytes")
    return tuple(b) if b is not None else tuple(entry["token"].encode("utf-8"))


def positions(reply):
    """[(token key, {token key: logprob})] from an OpenAI chat reply's choices[0].logprobs.content, [] without
    logprobs; keys are token_key's (bytes), so two ids with one text stay apart. A null logprob (one the server could
    not give) is left out; two listed ids with the same bytes (none in a byte-level vocabulary) are summed."""
    try:
        content = reply["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for c in content or []:
        top = {}
        for t in c.get("top_logprobs") or []:
            if t.get("logprob") is None:
                continue
            k, lp = token_key(t), float(t["logprob"])
            top[k] = lp if k not in top else math.log(math.exp(top[k]) + math.exp(lp))
        k = token_key(c)
        if k not in top and c.get("logprob") is not None:
            top[k] = float(c["logprob"])
        out.append((k, top))
    return out


def reply_text(reply):
    try:
        return reply["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


# -- the statistics ---------------------------------------------------------------------------------------------

def coarse_kl(p_top, q_top, eps=1e-10):
    """KL(P || Q) in nats over the shared top-K support plus one remainder bucket, from two top-K logprob maps: the
    tokens both list keep their probabilities, and each side's mass outside them (1 minus its shared mass: its other
    listed tokens and its unlisted tail) is one bucket. That is the exact KL of P and Q coarsened by the same partition,
    so never above the full-vocabulary KL (the data-processing inequality); 0 when the shared tokens carry the same
    probabilities on both sides."""
    shared = [t for t in p_top if t in q_top]
    p = {t: math.exp(p_top[t]) for t in shared}
    q = {t: math.exp(q_top[t]) for t in shared}
    p_rest = max(0.0, 1.0 - sum(p.values()))
    q_rest = max(0.0, 1.0 - sum(q.values()))
    kl = 0.0
    for t in shared:
        if p[t] > 0:
            kl += p[t] * math.log(p[t] / max(q[t], eps))
    if p_rest > 0:
        kl += p_rest * math.log(p_rest / max(q_rest, eps))
    return max(kl, 0.0)


def compare(base, masked):
    """One prompt's comparison from both endpoints' greedy positions [(token, top)] (or token lists without tops)."""
    n = max(len(base), len(masked))
    bt = [x[0] if isinstance(x, tuple) else x for x in base]
    mt = [x[0] if isinstance(x, tuple) else x for x in masked]
    agree = sum(1 for i in range(min(len(bt), len(mt))) if bt[i] == mt[i])
    prefix = 0
    while prefix < min(len(bt), len(mt)) and bt[prefix] == mt[prefix]:
        prefix += 1
    out = {"positions": n, "agree": agree, "agreement": (agree / n) if n else 1.0, "prefix": prefix,
           "exact": bt == mt, "kl_first": None, "kl_matched": []}
    have = base and masked and isinstance(base[0], tuple) and isinstance(masked[0], tuple) and base[0][1] and masked[0][1]
    if have:
        # the context is identical at positions 0..prefix (position prefix is where the greedy tokens part, if they do)
        last = min(prefix, len(base) - 1, len(masked) - 1)
        out["kl_matched"] = [coarse_kl(base[i][1], masked[i][1]) for i in range(last + 1)]
        out["kl_first"] = out["kl_matched"][0]
    return out


def p95(xs):
    """The 95th percentile by linear interpolation between closest ranks (numpy's default); None for no values."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * 0.95
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(rows):
    def block(rs):
        firsts = [r["kl_first"] for r in rs if r["kl_first"] is not None]
        pooled = [k for r in rs for k in r["kl_matched"]]
        return {"prompts": len(rs), "with_kl": len(firsts),
                "kl_first_mean": mean(firsts), "kl_first_p95": p95(firsts),
                "kl_matched_mean": mean(pooled), "kl_matched_p95": p95(pooled), "kl_matched_positions": len(pooled),
                "agreement_mean": mean([r["agreement"] for r in rs]), "prefix_mean": mean([r["prefix"] for r in rs]),
                "exact_share": mean([1.0 if r["exact"] else 0.0 for r in rs]),
                "first_token_agree_share": mean([1.0 if r["prefix"] > 0 else 0.0 for r in rs if r["positions"]])}
    doms = sorted({r["domain"] for r in rows})
    return {"all": block(rows), "domains": {d: block([r for r in rows if r["domain"] == d]) for d in doms}}


# -- kl ---------------------------------------------------------------------------------------------------------

def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Endpoint:
    def __init__(self, url, model, timeout, top_logprobs, extra):
        self.url = url.rstrip("/")
        self.model, self.timeout, self.top, self.extra = model, timeout, top_logprobs, extra
        self.logprobs = bool(top_logprobs)
        self.kl_status = "ok" if self.logprobs else "logprobs not asked (--top-logprobs 0)"

    def drop_logprobs(self, why):
        if self.logprobs:
            self.logprobs = False
            self.kl_status = why

    def ask(self, prompt, max_tokens):
        """(positions [(token, top)], reply) for one greedy request; [] without logprobs. A refusal drops logprobs for
        the rest of the run (asked again without them)."""
        st, rep = post_json(self.url + "/v1/chat/completions", chat_body(prompt, self.model, max_tokens, self.top if self.logprobs else 0, self.extra), self.timeout)
        if self.logprobs and refuses_logprobs(st, rep):
            self.drop_logprobs("server refuses logprobs (%s): %s" % (self.url, json.dumps(rep)[:200]))
            st, rep = post_json(self.url + "/v1/chat/completions", chat_body(prompt, self.model, max_tokens, 0, self.extra), self.timeout)
        if st != 200 or not isinstance(rep, dict):
            raise RuntimeError("%s: HTTP %s: %s" % (self.url, st, str(rep)[:300]))
        pos = positions(rep) if self.logprobs else []
        if self.logprobs and not pos and reply_text(rep):
            self.drop_logprobs("server answered without logprobs (%s)" % self.url)
        return pos, rep

    def tokens(self, rep, max_tokens):
        """The reply's token ids by the endpoint's /tokenize (when the reply carries no logprobs)."""
        ts, tk = post_json(self.url + "/tokenize", {"model": self.model, "prompt": reply_text(rep), "add_special_tokens": False}, self.timeout)
        if ts != 200 or not isinstance(tk, dict) or "tokens" not in tk:
            raise RuntimeError("%s/tokenize: HTTP %s: %s" % (self.url, ts, str(tk)[:300]))
        return list(tk["tokens"])[:max_tokens]


def run_kl(a):
    prompts = load_jsonl(a.prompts)
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise SystemExit("REFUSED: duplicate prompt ids in %s" % a.prompts)
    mask_info = load_mask(a.mask)[1] if a.mask else None
    extra = json.loads(a.body_json) if a.body_json else {}
    base = Endpoint(a.base, a.model, a.timeout, a.top_logprobs, extra)
    masked = Endpoint(a.masked, a.model, a.timeout, a.top_logprobs, extra)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    with open(os.path.join(a.out, "kl-rows.jsonl"), "w", encoding="utf-8") as f:
        for p in prompts:
            bp, brep = base.ask(p, a.positions)
            mp, mrep = masked.ask(p, a.positions)
            if not (base.logprobs and masked.logprobs):
                # one side without logprobs: neither side's KL, and both sides' tokens by /tokenize (alike)
                why = base.kl_status if not base.logprobs else masked.kl_status
                base.drop_logprobs(why)
                masked.drop_logprobs(why)
                c = compare(base.tokens(brep, a.positions), masked.tokens(mrep, a.positions))
            else:
                c = compare(bp, mp)
            row = {"id": p["id"], "domain": p.get("domain", "default"), **c,
                   "base_text_sha256": hashlib.sha256(reply_text(brep).encode()).hexdigest(),
                   "masked_text_sha256": hashlib.sha256(reply_text(mrep).encode()).hexdigest()}
            rows.append(row)
            f.write(json.dumps(row, sort_keys=True) + "\n")
    prompts_sha = hashlib.sha256(read_bytes(a.prompts)).hexdigest()
    kl_ok = base.logprobs and masked.logprobs
    summary = {"tool": "ablate.py kl (pod card 5bb471d3b9)", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "base_url": a.base, "masked_url": a.masked, "mask": mask_info, "prompts": os.path.abspath(a.prompts),
               "prompts_sha256": prompts_sha, "positions": a.positions, "top_logprobs": a.top_logprobs,
               "kl_status": "ok" if kl_ok else (base.kl_status if not base.logprobs else masked.kl_status),
               "kl_definition": "KL(unmasked || masked) in nats over the top-K tokens both endpoints list plus one remainder bucket (a lower bound on the full-vocabulary KL)",
               **summarize(rows)}
    if a.require_kl and not kl_ok:
        dump_json(os.path.join(a.out, "kl-summary.json"), summary)
        raise SystemExit("REFUSED: --require-kl and %s" % summary["kl_status"])
    dump_json(os.path.join(a.out, "kl-summary.json"), summary)
    with open(os.path.join(a.out, "kl-summary.md"), "w", encoding="utf-8") as f:
        f.write(kl_markdown(summary))
    return 0


def _f(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, x))


def kl_markdown(s):
    m = s.get("mask") or {}
    lines = ["# Expert-mask KL: %s" % (("mask=" + m["id"]) if m.get("id") else "no mask file named"), "",
             "unmasked %s vs masked %s; %d positions greedy, top-%d logprobs; kl_status: %s" % (s["base_url"], s["masked_url"], s["positions"], s["top_logprobs"], s["kl_status"]),
             "prompts %s (sha256 %s)" % (s["prompts"], s["prompts_sha256"][:16]), "",
             "| domain | prompts | KL first mean | KL first p95 | KL matched mean | KL matched p95 | agreement | prefix | exact |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, b in [("all", s["all"])] + sorted(s["domains"].items()):
        lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (name, b["prompts"], _f(b["kl_first_mean"]), _f(b["kl_first_p95"]), _f(b["kl_matched_mean"]), _f(b["kl_matched_p95"]), _f(b["agreement_mean"], 3), _f(b["prefix_mean"], 1), _f(b["exact_share"], 3)))
    return "\n".join(lines) + "\n"


# -- gate -------------------------------------------------------------------------------------------------------

def gate_arms(gate_path, engine):
    with open(gate_path, "rb") as f:
        g = tomllib.load(f)
    if engine not in (g.get("engines") or {}):
        raise SystemExit("REFUSED: engine %s is not in %s" % (engine, gate_path))
    return [x["name"] for x in g.get("arms", []) if x.get("engine") == engine]


def run_harness(a, out, url, arms, log):
    cmd = [a.python, os.path.join(a.eval_repo, "tools", "eval", "run.py"), "--gate", a.gate, "--tasks", a.tasks,
           "--manifest", a.manifest, "--out", out, "--engine", "%s=%s" % (a.engine, url), "--arms", ",".join(arms),
           "--split", a.split]
    for k, v in (("--suites", a.suites), ("--deadline-utc", a.deadline_utc)):
        if v:
            cmd += [k, v]
    for d in a.draws:
        cmd += ["--draws", d]
    if a.limit_tasks:
        cmd += ["--limit-tasks", str(a.limit_tasks)]
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    if a.dry_run:
        return 0
    return subprocess.run(cmd, cwd=a.eval_repo, stdout=log, stderr=subprocess.STDOUT).returncode


def run_analyze(a, run_dir, log):
    cmd = [a.python, os.path.join(a.eval_repo, "tools", "eval", "analyze.py"), "--gate", a.gate, "--tasks", a.tasks,
           "--manifest", a.manifest, "--run", run_dir, "--out", os.path.join(run_dir, "analysis")]
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    return 0 if a.dry_run else subprocess.run(cmd, cwd=a.eval_repo, stdout=log, stderr=subprocess.STDOUT).returncode


def scored_rows(run_dir):
    """{(task, arm, draw): row} of a run's replies.jsonl, a scored row winning over an unscored one (load_rows)."""
    rows = {}
    p = os.path.join(run_dir, "replies.jsonl")
    for r in (load_jsonl(p) if os.path.exists(p) else []):
        k = (r["task"], r["arm"], int(r["draw"]))
        if r.get("status") in SCORED or k not in rows or rows[k].get("status") not in SCORED:
            rows[k] = r
    return {k: r for k, r in rows.items() if r.get("status") in SCORED}


def correct(r):
    return bool((r.get("score") or {}).get("correct"))


def cluster_ci(pairs, families, resamples, seed):
    """95 % interval of the mean paired difference (masked - unmasked), families resampled with replacement."""
    by = {}
    for (t, d), (b, m) in pairs.items():
        by.setdefault(families.get(t, t), []).append(m - b)
    fams = sorted(by)
    if len(fams) < 2:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        tot = n = 0
        for _ in fams:
            xs = by[fams[rng.randrange(len(fams))]]
            tot += sum(xs)
            n += len(xs)
        stats.append(tot / n)
    stats.sort()
    return stats[int(0.025 * (resamples - 1))], stats[int(math.ceil(0.975 * (resamples - 1)))]


def deltas(base_dir, masked_dir, tasks_path, resamples=2000, seed=0x5BB471D3B9):
    tasks = {t["id"]: t for t in load_jsonl(tasks_path)}
    families = {i: t.get("family", i) for i, t in tasks.items()}
    b, m = scored_rows(base_dir), scored_rows(masked_dir)
    out = []
    keys = sorted({(tasks.get(t, {}).get("suite", "?"), arm) for (t, arm, _) in set(b) | set(m)})
    for suite, arm in keys:
        bk = {k for k in b if k[1] == arm and tasks.get(k[0], {}).get("suite", "?") == suite}
        mk = {k for k in m if k[1] == arm and tasks.get(k[0], {}).get("suite", "?") == suite}
        both = sorted(bk & mk)
        pairs = {(k[0], k[2]): (1 if correct(b[k]) else 0, 1 if correct(m[k]) else 0) for k in both}
        n = len(pairs)
        bacc = mean([v[0] for v in pairs.values()])
        macc = mean([v[1] for v in pairs.values()])
        lo, hi = cluster_ci(pairs, families, resamples, seed)
        out.append({"suite": suite, "arm": arm, "pairs": n, "families": len({families.get(t, t) for (t, _) in pairs}),
                    "acc_unmasked": bacc, "acc_masked": macc, "delta": (macc - bacc) if n else None,
                    "ci95": [lo, hi], "lost": sum(1 for v in pairs.values() if v == (1, 0)),
                    "gained": sum(1 for v in pairs.values() if v == (0, 1)),
                    "unpaired_unmasked": len(bk - mk), "unpaired_masked": len(mk - bk)})
    return out


def deltas_markdown(rep):
    m = rep["mask"]
    lines = ["# Expert-mask ablation: mask=%s (%d experts in %d layers)" % (m["id"] or "(empty)", m["experts_masked"], len(m["layers"])), "",
             "gate %s; engine %s unmasked %s vs masked %s (%s)" % (rep["gate"], rep["engine"], rep["base"], rep["masked"], m["path"]),
             "delta = masked - unmasked accuracy over (task, draw) pairs both scored; 95 %% CI: family-cluster bootstrap, %d resamples" % rep["resamples"], "",
             "| suite | arm | pairs | families | unmasked | masked | delta | 95 % CI | lost | gained | unpaired u/m |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["deltas"]:
        lines.append("| %s | %s | %d | %d | %s | %s | %s | [%s, %s] | %d | %d | %d/%d |" % (
            r["suite"], r["arm"], r["pairs"], r["families"], _f(r["acc_unmasked"], 3), _f(r["acc_masked"], 3), _f(r["delta"], 3),
            _f(r["ci95"][0], 3), _f(r["ci95"][1], 3), r["lost"], r["gained"], r["unpaired_unmasked"], r["unpaired_masked"]))
    return "\n".join(lines) + "\n"


def run_gate(a):
    mask, info = load_mask(a.mask)
    if not info["id"] and not a.allow_empty:
        raise SystemExit("REFUSED: %s masks nothing (an A/A run needs --allow-empty)" % a.mask)
    if not a.base_url and not a.base_run:
        raise SystemExit("REFUSED: name the unmasked side: --base-url URL or --base-run DIR")
    arms = a.arms.split(",") if a.arms else gate_arms(a.gate, a.engine)
    if not arms:
        raise SystemExit("REFUSED: no arm of %s runs on engine %s" % (a.gate, a.engine))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ablate.log"), "a", encoding="utf-8") as log:
        return _run_gate(a, info, arms, log)


def _run_gate(a, info, arms, log):
    gate_sha = hashlib.sha256(read_bytes(a.gate)).hexdigest()
    masked_dir = os.path.join(a.out, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    mark = {"tool": "ablate.py gate (pod card 5bb471d3b9)", "masked": True, "engine": a.engine, "url": a.masked_url,
            "mask": info, "gate": os.path.abspath(a.gate), "gate_sha256": gate_sha, "arms": arms,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dump_json(os.path.join(masked_dir, "ablation.json"), mark)
    rc = run_harness(a, masked_dir, a.masked_url, arms, log)
    if rc != 0:
        raise SystemExit("the masked run's run.py exited %d (see %s)" % (rc, os.path.join(a.out, "ablate.log")))
    if a.base_run:
        base_dir = a.base_run
    else:
        base_dir = os.path.join(a.out, "base")
        os.makedirs(base_dir, exist_ok=True)
        dump_json(os.path.join(base_dir, "ablation.json"), dict(mark, masked=False, url=a.base_url, mask=None))
        rc = run_harness(a, base_dir, a.base_url, arms, log)
        if rc != 0:
            raise SystemExit("the unmasked run's run.py exited %d (see %s)" % (rc, os.path.join(a.out, "ablate.log")))
    if a.analyze:
        for d in (masked_dir, base_dir):
            run_analyze(a, d, log)
    if a.dry_run:
        return 0
    rep = {"tool": mark["tool"], "gate": os.path.abspath(a.gate), "gate_sha256": gate_sha, "engine": a.engine,
           "base": base_dir if a.base_run else a.base_url, "masked": a.masked_url, "mask": info, "arms": arms,
           "resamples": a.resamples, "deltas": deltas(base_dir, masked_dir, a.tasks, a.resamples)}
    dump_json(os.path.join(a.out, "deltas.json"), rep)
    with open(os.path.join(a.out, "deltas.md"), "w", encoding="utf-8") as f:
        f.write(deltas_markdown(rep))
    return 0


def run_id(a):
    mask, info = load_mask(a.mask)
    print(json.dumps(info, sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kl", help="KL and token agreement on a calibration prompt set")
    k.add_argument("--base", required=True, help="the unmasked endpoint's base URL")
    k.add_argument("--masked", required=True, help="the masked endpoint's base URL")
    k.add_argument("--prompts", required=True, help="JSONL: id, domain, messages[, body]")
    k.add_argument("--mask", help="the mask file the masked endpoint runs (named in the summary by its id)")
    k.add_argument("--out", required=True)
    k.add_argument("--model", default="dsv41")
    k.add_argument("--positions", type=int, default=16, help="greedy tokens a prompt (max_tokens)")
    k.add_argument("--top-logprobs", type=int, default=20)
    k.add_argument("--timeout", type=float, default=600.0)
    k.add_argument("--body-json", help="extra request fields for every prompt, as JSON")
    k.add_argument("--require-kl", action="store_true", help="exit non-zero when an endpoint refuses logprobs")
    g = sub.add_parser("gate", help="an FQ-1 gate run masked and unmasked through the eval harness, with score deltas")
    g.add_argument("--eval-repo", required=True, help="the checkout holding tools/eval/run.py")
    g.add_argument("--gate", required=True)
    g.add_argument("--tasks", required=True)
    g.add_argument("--manifest", required=True)
    g.add_argument("--engine", required=True, help="the gate's engine both runs use (e.g. pair)")
    g.add_argument("--masked-url", required=True)
    g.add_argument("--base-url", help="the unmasked endpoint (run here into OUT/base)")
    g.add_argument("--base-run", help="an earlier unmasked run directory of the same gate and arms")
    g.add_argument("--mask", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--arms", help="comma list (default: every arm of the gate on --engine)")
    g.add_argument("--suites")
    g.add_argument("--split", default="test")
    g.add_argument("--draws", action="append", default=[])
    g.add_argument("--deadline-utc")
    g.add_argument("--limit-tasks", type=int, default=0)
    g.add_argument("--analyze", action="store_true")
    g.add_argument("--allow-empty", action="store_true")
    g.add_argument("--resamples", type=int, default=2000)
    g.add_argument("--python", default=sys.executable)
    g.add_argument("--dry-run", action="store_true", help="write the marks and log the commands, run nothing")
    i = sub.add_parser("id", help="check a mask file and print its canonical id")
    i.add_argument("--mask", required=True)
    a = ap.parse_args(argv)
    try:
        return {"kl": run_kl, "gate": run_gate, "id": run_id}[a.cmd](a)
    except MaskError as e:
        print("REFUSED: mask: %s" % e, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
