"""The lane tool (tf-dsv41-lanes) on a pair, as serving runs it, and its --profile 2 output as a table.

Why the scratch lane_rank.sh: the StrixFold runtime dlopens "libamdhip64.so.7" by bare name (gpu_facade hip/driver.zig
Driver.open); the workers have no ld.so.cache entry for it, so it is found only through the LD_LIBRARY_PATH that the
serving conf's WRAP0/WRAP1 put in front of the rank (serve_lib.sh serve_spawn: env -i SERVE_ENV... WRAP... SERVE_CMD).
The deployment's qual/lane_rank.sh execs without the WRAP, so the lane tool stops at DriverUnavailable. Until its owner
fixes it upstream, a scratch tools directory (symlinks to the deployment's tools/dsv41-hip, the deployment itself never
touched) holds a copy of lane_rank.sh whose exec line applies the rank's WRAP exactly as serve_spawn does and records it
in rankR.cmd ("wrap (WRAPR, as serve_spawn): ...").

--profile 2 (catalog/strixfold executor dsv41 lanes_main.zig, round.zig PhaseTimer): each rank prints
  {"rank": R, "prefill_phases_ms": {phase: mean ms a prompt chunk, ...}, "chunks": N}   (both ranks, at exit)
rank 0 also {"rank": 0, "phases_ms": {phase: mean ms a round}} for the decode rounds (eager rounds only: under
GRAPHS=1 the rounds run from graphs and are not timed) and {"rank": 0, "profile": {"rounds": ..., ...}}; rank 1
{"rank": 1, "follow_profile": {...}}. A phase's time runs from the mark before it to its own, so the phases (the
coarse ones and the a_* / m_* sub-phases inside attention and the MoE) are disjoint and sum to the chunk's or round's
GPU time. The means are over all chunks / timed rounds: the binary keeps no per-chunk rows.
"""
import json
import shlex

EXEC_LINE = 'exec "${pin[@]}" env -i "${SERVE_ENV[@]}" "${SERVE_CMD[@]}" > "rank$R.jsonl" 2> "rank$R.err"'
WRAP_EXEC = """# strixflerp scratch copy: the rank's WRAP applied as serve_lib.sh serve_spawn applies it (wv=WRAP$RANK)
wv=WRAP$RANK
wrap=()
[ -n "${!wv:-}" ] && read -ra wrap <<< "${!wv}"
echo "wrap ($wv, as serve_spawn): ${wrap[*]:-(none)}" >> "rank$R.cmd"
exec "${pin[@]}" env -i "${SERVE_ENV[@]}" "${wrap[@]}" "${SERVE_CMD[@]}" > "rank$R.jsonl" 2> "rank$R.err\""""
WRAP_MARK = "wrap ($wv, as serve_spawn)"
LANE_RANK_SHA = "c1ea5a694ce3e95d96cbd0ce0d4dbc5ed06cf89e9f80f78ad9754cce93eaf461"  # f64b5ff2e28b's, read 2026-11-11


def patch_lane_rank(text):
    """The deployment's lane_rank.sh with its one exec line replaced by WRAP_EXEC; ValueError if not exactly once."""
    n = text.count(EXEC_LINE)
    if n != 1:
        raise ValueError(f"lane_rank.sh: the exec line occurs {n} times, not once (the upstream script changed)")
    return text.replace(EXEC_LINE, WRAP_EXEC)


def stage_cmd(deploy, tools):
    """Shell for a worker: TOOLS/dsv41-hip mirroring DEPLOY/tools/dsv41-hip by symlinks, qual/ a directory of symlinks
    without lane_rank.sh (the patched copy is put there next)."""
    src = f"{deploy}/tools/dsv41-hip"
    return (f"rm -rf {tools} && mkdir -p {tools}/dsv41-hip/qual && "
            f"for f in {src}/*; do b=$(basename $f); [ $b = qual ] || ln -s $f {tools}/dsv41-hip/$b; done && "
            f"for f in {src}/qual/*; do b=$(basename $f); [ $b = lane_rank.sh ] || ln -s $f {tools}/dsv41-hip/qual/$b; done && "
            f"test -L {tools}/dsv41-hip/serve_lib.sh && test -L {tools}/dsv41-hip/pidexec.sh")


def records(text):
    """The JSON lines of a rank's stdout (other lines skipped)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def parse(rank_texts):
    """{rank: stdout text} -> {"prefill": {rank: {...}}, "decode": {...} or None, "requests": [...], "summary": ...}."""
    doc = {"prefill": {}, "decode": None, "requests": [], "summary": None, "profile": {}}
    for rank, text in sorted(rank_texts.items()):
        for r in records(text):
            if "prefill_phases_ms" in r:
                doc["prefill"][rank] = {"chunks": r.get("chunks", 0), "ms": r["prefill_phases_ms"]}
            elif "phases_ms" in r:
                doc["decode"] = {"rank": rank, "ms": r["phases_ms"]}
            elif "profile" in r or "follow_profile" in r:
                doc["profile"][rank] = r.get("profile") or r.get("follow_profile")
            elif "request" in r:
                doc["requests"].append(r)
            elif "requests" in r and "steps" in r:
                doc["summary"] = r
    if doc["decode"] is not None:
        p = doc["profile"].get(doc["decode"]["rank"]) or {}
        doc["decode"]["rounds"] = p.get("rounds")
    return doc


def _rows(ms, scale=1):
    total = sum(v for v in ms.values() if v)
    rows = [(k, v, v * scale, (100.0 * v / total) if total else 0.0) for k, v in ms.items() if v]
    return rows, total


def table(doc):
    """Markdown: the prefill table a rank (ms a chunk, ms over the chunks, share) and the decode table (rank 0)."""
    out = []
    for rank, p in sorted(doc["prefill"].items()):
        n = p["chunks"] or 0
        rows, total = _rows(p["ms"], n)
        out += [f"### prefill, rank {rank}: {n} chunks, {total:.2f} ms GPU a chunk, {total * n / 1e3:.2f} s over the prompt", "",
                "| phase | ms a chunk | ms over the prompt | share |", "|---|---:|---:|---:|"]
        out += [f"| {k} | {v:.3f} | {t:.1f} | {s:.1f}% |" for k, v, t, s in rows] + [""]
    d = doc["decode"]
    if d:
        rows, total = _rows(d["ms"])
        rounds = d.get("rounds")
        out += [f"### decode, rank {d['rank']}: {total:.2f} ms GPU a timed round"
                + (f" ({rounds} rounds)" if rounds is not None else "")
                + ("" if total else " -- no timed rounds (GRAPHS=1 rounds are not timed)"), "",
                "| phase | ms a round | share |", "|---|---:|---:|"]
        out += [f"| {k} | {v:.3f} | {s:.1f}% |" for k, v, _, s in rows] + [""]
    else:
        out += ["### decode: no phases_ms line (rank 0 prints it only with --profile 2)", ""]
    for r in doc["requests"]:
        if "tokens" in r:
            out.append(f"request {r.get('name')}: {r['tokens']} tokens in {r.get('seconds')} s, {r.get('rounds')} rounds, "
                       f"{r.get('accepted')}/{r.get('drafted')} drafts accepted")
        elif "prefill_s" in r:
            out.append(f"request {r.get('name')}: {r.get('prompt_tokens')} prompt tokens, prefill {r['prefill_s']} s")
    return "\n".join(out).rstrip() + "\n"


def stage(run, ips, deploy, tools, ssh, out_dir):
    """The scratch tools on each worker (lane_rank.sh read from the first, the same sha required on every worker);
    the patched copy also saved to OUT_DIR as evidence. Returns an error text or None."""
    import os
    rc, text = run.on(ips[0], f"cat {deploy}/tools/dsv41-hip/qual/lane_rank.sh", 30, True)
    if rc != 0:
        return f"could not read lane_rank.sh on {ips[0]}"
    try:
        patched = patch_lane_rank(text + "\n")
    except ValueError as e:
        return str(e)
    local = os.path.join(out_dir, "lane_rank.scratch.sh")
    with open(local, "w") as f:
        f.write(patched)
    os.chmod(local, 0o755)
    for ip in ips:
        rc, sha = run.on(ip, f"sha256sum < {deploy}/tools/dsv41-hip/qual/lane_rank.sh | cut -c1-64", 30, True)
        if sha.strip() != LANE_RANK_SHA:
            run.say(f"note: lane_rank.sh on {ip} is {sha.strip()[:12]}, not the {LANE_RANK_SHA[:12]} read 2026-11-11 (the patch still applied once)")
        if run.on(ip, stage_cmd(deploy, tools), 60)[0] != 0:
            return f"could not stage the scratch tools on {ip}"
        if run.sh(["rsync", "-a", "-e", " ".join(ssh), local, f"pod@{ip}:{tools}/dsv41-hip/qual/lane_rank.sh"], 60)[0] != 0:
            return f"could not copy the patched lane_rank.sh to {ip}"
        if run.on(ip, f"grep -cF {shlex.quote(WRAP_MARK)} {tools}/dsv41-hip/qual/lane_rank.sh", 30)[1].strip() != "1":
            return f"the patched lane_rank.sh on {ip} is not the staged one"
    return None


def run_lanes(run, r0, r1, tools, run_dir, conf, requests_remote, xdg, wait_s=300, units=("strixflerp-prefill-r0", "strixflerp-prefill-r1")):
    """tf-dsv41-lanes --profile 2 on both ranks through the scratch lane_rank.sh (rank 1 first); waits for rank 0 to
    end (at most WAIT_S), stops both, returns {"seconds", "texts": {(rank, ext): text}, "wrap": {rank: line}}."""
    import shlex
    import time
    q = f"cd {tools}/dsv41-hip/qual && ./lane_rank.sh {conf}"
    run.on(r1, xdg + f"systemd-run --user --collect --unit={units[1]} --property=KillMode=process bash -c "
           + shlex.quote(f"{q} 1 {run_dir} --profile 2"))
    time.sleep(3)
    run.on(r0, xdg + f"systemd-run --user --collect --unit={units[0]} --property=KillMode=process bash -c "
           + shlex.quote(f"{q} 0 {run_dir} --requests {requests_remote} --parallel 1 --logits-sha 1 --profile 2"))
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(5)
        if run.on(r0, xdg + f"systemctl --user is-active -q {units[0]}", 20, True)[0] != 0:
            break
    row = {"seconds": round(time.time() - t0, 1), "texts": {}, "wrap": {}}
    time.sleep(3)
    for ip, r in ((r0, 0), (r1, 1)):
        run.on(ip, xdg + f"systemctl --user stop {units[r]} 2>/dev/null; true", 60)
        for ext in ("jsonl", "err", "cmd"):
            _, text = run.on(ip, f"cat {run_dir}/rank{r}.{ext} 2>/dev/null | cut -c1-6000", 60, True)
            row["texts"][(r, ext)] = text
        row["wrap"][r] = next((l for l in row["texts"][(r, "cmd")].splitlines() if l.startswith("wrap (")), None)
    return row
