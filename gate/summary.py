"""The one summary writer of the gate scripts (card 6249b8db6f, owner's review finding 8).

Thirteen gate scripts each had a write_summary; they render thirteen different reports, and what they shared was the
mechanics: summary.json as json.dump with indent 1 and a newline, summary.md as lines joined with a trailing newline,
the all-stages-pass verdict and the per-stage PASS/FAIL bullet. Those live here; the report bodies stay in the scripts.

Provenance (agreed with agent, card a1fe34d5c6): every JSON document written here carries `config_sha256`, the
sha256 of the gate TOML the script ran under, so a result can be tied to the criteria it was judged by. The path is
given (`config=`) or discovered from the script's ctx: ctx["config"] (the moe_* scripts), ctx["recipe"].gate["config"]
(the recipe gates), ctx["config_rel"] (strixite_prefill). A script with no TOML (gguf_census, replay) writes null.
"""
import hashlib
import json
import os
import socket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = "config_sha256"


def host():
    return socket.gethostname()


def _resolve(p):
    """A relative path is the scripts' convention: relative to the tree root (recipe.gate["config"], --config defaults)."""
    if not p:
        return None
    if os.path.isabs(p):
        return p
    q = os.path.join(ROOT, p)
    return q if os.path.isfile(q) or not os.path.isfile(p) else os.path.abspath(p)


def config_path(ctx):
    """The gate TOML a script's ctx names, or None."""
    if not ctx:
        return None
    if ctx.get("config"):
        return _resolve(ctx["config"])
    recipe = ctx.get("recipe")
    gate = getattr(recipe, "gate", None)
    if isinstance(gate, dict) and gate.get("config"):
        return _resolve(gate["config"])
    if ctx.get("config_rel"):
        return _resolve(ctx["config_rel"])
    return None


def config_sha256(path):
    """sha256 of the file, or None when there is no file."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(directory, doc, name="summary.json", config=None, ctx=None):
    """json.dump(doc, indent=1) plus a newline, with `config_sha256` added as the last key; returns the document.
    The key is always present (null when the script has no TOML) so a reader can tell "no config" from "old writer"."""
    doc = dict(doc)
    doc[FIELD] = config_sha256(config if config is not None else config_path(ctx))
    with open(os.path.join(directory, name), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return doc


def write_md(directory, lines, name="summary.md", newline=True):
    """The lines joined; a trailing newline unless a script's recorded format has none (replay)."""
    text = "\n".join(lines) + ("\n" if newline else "")
    with open(os.path.join(directory, name), "w") as f:
        f.write(text)
    return text


def all_pass(stages, results):
    return all((results.get(s) or {}).get("pass") for s in stages)


def verdict(stages, results):
    """PASSED, NOT COMPLETE: <stages> not run (every stage that ran passed), or FAILED."""
    if all_pass(stages, results):
        return "PASSED"
    missing = [s for s in stages if s not in results]
    if missing and all(results[s].get("pass") for s in results):
        return "NOT COMPLETE: " + ", ".join(missing) + " not run"
    return "FAILED"


def stage_bullets(stages, results):
    """- **stage**: PASS|FAIL [- blocked] [- error], for the stages that ran, in order."""
    out = []
    for s in stages:
        if s in results:
            r = results[s]
            out.append(f"- **{s}**: {'PASS' if r.get('pass') else 'FAIL'}" + (f" - {r['blocked']}" if r.get("blocked") else "")
                       + (f" - {r['error']}" if r.get("error") else ""))
    return out
