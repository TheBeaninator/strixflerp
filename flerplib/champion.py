"""The champion ledger (configs/champions.toml, rendered to docs/CHAMPIONS.md): per area, the configuration that holds
the best number that passed the area's fidelity floor, with its evidence; earlier champions are kept as history.

    flerp champion show [AREA]
    flerp champion propose CONFIG AREA RESULTS_DIR [--case C] [--dry-run]

A results directory carries champion.json, written by the gate that produced it (write_result):
    {"config": NAME, "area": AREA, "case": "", "value": N, "run_id": ID, "date": "YYYY-MM-DD",
     "fidelity": {"identical": true} | {"kl_vs_fp32": X}, "scale": "397B" | "35B proxy", "note": "..."}
`propose` promotes only when the config name, area and case match, the value beats the incumbent in the area's direction,
and the fidelity result passes the area's floor. A promotion moves the incumbent to history and rewrites both files;
nothing else is ever overwritten. A refusal says why and exits 1."""
import datetime
import json
import os
import sys
import tomllib

from . import tomlw
from .configs import ROOT

LEDGER = os.path.join(ROOT, "configs", "champions.toml")
DOC = os.path.join(ROOT, "docs", "CHAMPIONS.md")


def load_ledger(path=LEDGER):
    with open(path, "rb") as f:
        d = tomllib.load(f)
    return d.get("area", {}), d.get("champion", [])


def save_ledger(areas, champs, path=LEDGER, doc=DOC):
    head = ("# Champion ledger (see docs/CHAMPIONS.md and flerp champion --help). Edited by `flerp champion propose`, "
            "not by hand.\n\n")
    body = tomlw.dump([(f"area.{k}", v) for k, v in areas.items()] + [("[champion]", champs)])
    with open(path, "w") as f:
        f.write(head + body)
    with open(doc, "w") as f:
        f.write(render(areas, champs))


def _key(c):
    return (c["area"], c.get("case", ""))


def current(champs, area, case=""):
    for c in champs:
        if c["area"] == area and c.get("case", "") == case and c["status"] == "current":
            return c
    return None


def better(area, new, old):
    return new > old if area["direction"] == "higher" else new < old


def floor_check(area, fid):
    """-> (ok, text) for a fidelity record against the area's floor."""
    kind = area["fidelity"]
    if kind == "identity":
        ok = fid.get("identical") is True
        return ok, "identical ids" if ok else f"identity not shown ({fid})"
    kl = fid.get("kl_vs_fp32")
    if kl is None:
        return False, "no kl_vs_fp32 recorded"
    ok = kl <= area["fidelity_bound"]
    return ok, f"KL vs FP32 {kl} {'<=' if ok else '>'} {area['fidelity_bound']}"


def write_result(directory, config, area, value, run_id, fidelity, case="", scale="397B", note="", date=None):
    """What a gate calls to record its config name and number (champion.json) next to its other results."""
    os.makedirs(directory, exist_ok=True)
    doc = {"config": config, "area": area, "case": case, "value": value, "run_id": run_id, "fidelity": fidelity,
           "scale": scale, "note": note, "date": date or datetime.date.today().isoformat()}
    with open(os.path.join(directory, "champion.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def propose(config, area_name, results_dir, case="", dry=False, ledger=LEDGER, doc=DOC):
    areas, champs = load_ledger(ledger)
    if area_name not in areas:
        return False, f"unknown area {area_name!r} ({', '.join(areas)})"
    area = areas[area_name]
    p = os.path.join(results_dir, "champion.json")
    if not os.path.isfile(p):
        return False, f"{p} missing: the gate must record its config, value and fidelity (champion.write_result)"
    with open(p) as f:
        r = json.load(f)
    if (r.get("config"), r.get("area"), r.get("case", "")) != (config, area_name, case):
        return False, f"results say config {r.get('config')!r} area {r.get('area')!r} case {r.get('case', '')!r}, not {config!r} {area_name!r} {case!r}"
    if not os.path.isfile(os.path.join(ROOT, "configs", f"{config}.toml")):
        return False, f"no config file configs/{config}.toml"
    ok, why = floor_check(area, r.get("fidelity", {}))
    if not ok:
        return False, f"fidelity floor: {why}"
    inc = current(champs, area_name, case)
    if inc is not None and not better(area, r["value"], inc["value"]):
        return False, f"{r['value']} {area['unit']} does not beat the incumbent {inc['config']} {inc['value']} ({area['direction']} is better)"
    new = {"area": area_name, "case": case, "config": config, "value": r["value"], "unit": area["unit"],
           "scale": r.get("scale", "397B"), "run_id": str(r.get("run_id", "")), "evidence": results_dir,
           "fidelity": why, "date": r.get("date", ""), "status": "current", "note": r.get("note", "")}
    msg = f"{area_name}{' / ' + case if case else ''}: {config} {r['value']} {area['unit']} ({why})"
    if inc is not None:
        msg += f" replaces {inc['config']} {inc['value']}"
    if dry:
        return True, "would promote " + msg
    if inc is not None:
        inc["status"] = "history"
        inc["superseded_by"] = f"{config} {r.get('run_id', '')} {new['date']}"
    champs.append(new)
    save_ledger(areas, champs, ledger, doc)
    return True, "promoted " + msg


def render(areas, champs):
    L = ["# Champions", "",
         "Generated from `configs/champions.toml` by `flerp champion`; do not edit. Each area has one current champion, a",
         "named configuration (`configs/NAME.toml`) with the best number that passed the area's fidelity floor. A candidate",
         "is promoted with `flerp champion propose CONFIG AREA RESULTS_DIR` only when it beats the incumbent and passes the",
         "floor; the incumbent moves to history. `ornith397.serve` (decode-best + prefill-best) is what the engine launches.", ""]
    for name, a in areas.items():
        L += [f"## {name}", "", f"{a['description']}. Metric: {a['unit']}, {a['direction']} is better. Fidelity floor: {a['floor_text']}.", "",
              "| status | case | config | value | scale | fidelity | evidence (run) | date |", "|---|---|---|---|---|---|---|---|"]
        rows = sorted((c for c in champs if c["area"] == name), key=lambda c: (c.get("case", ""), c["status"] != "current", c["date"]))
        for c in rows:
            L.append(f"| {c['status']} | {c.get('case', '')} | `{c['config']}` | {c['value']} {c['unit']} | {c['scale']} | {c['fidelity']} | "
                     f"{c['evidence']}{' (' + c['run_id'] + ')' if c['run_id'] else ''} | {c['date']} |")
        notes = [f"- {c.get('case') or name} {c['config']}: {c['note']}" for c in rows if c.get("note")]
        L += [""] + notes + ([""] if notes else [])
    return "\n".join(L)


def cmd_champion(cat, a):
    if a.action == "show":
        areas, champs = load_ledger()
        for name, ar in areas.items():
            if a.args and name != a.args[0]:
                continue
            for c in champs:
                if c["area"] == name and c["status"] == "current":
                    print(f"{name:14} {c.get('case', ''):9} {c['config']:26} {c['value']:>9} {c['unit']:7} [{c['scale']}] {c['fidelity']}; {c['evidence']} ({c['run_id']}) {c['date']}")
        return 0
    if len(a.args) != 3:
        print("usage: flerp champion propose CONFIG AREA RESULTS_DIR [--case C] [--dry-run]", file=sys.stderr)
        return 2
    ok, msg = propose(a.args[0], a.args[1], a.args[2], a.case, a.dry_run)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1
