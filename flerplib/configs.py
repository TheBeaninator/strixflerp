"""Named runtime configurations (configs/NAME.toml): the recipe's settings a launcher applies on top of a build.
A config may include others (later ones and the file itself override earlier ones); `--set key=value` overrides
(stage keys bare, env keys as env.NAME) are the explicit way back to the old behaviour. See docs/CHAMPIONS.md."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_KEYS = {"rows": str, "reply": str, "drafts": int, "draft_head": bool, "with_mtp": bool, "context": int, "reuse_tail": int, "reuse_grid": int}


def _read(name, root):
    p = os.path.join(root, "configs", f"{name}.toml")
    if not os.path.isfile(p):
        raise SystemExit(f"no config {name!r} ({p})")
    with open(p, "rb") as f:
        return tomllib.load(f)


def load(name, root=ROOT, _seen=()):
    """-> {"name", "chain": [names, includes first], "stage": {...}, "env": {...}}. name 'none' = the old behaviour."""
    if name in (None, "", "none"):
        return {"name": "none", "chain": [], "stage": {}, "env": {}}
    if name in _seen:
        raise SystemExit(f"config include cycle at {name}")
    doc = _read(name, root)
    out = {"name": name, "chain": [], "stage": {}, "env": {}}
    for inc in doc.get("includes", []):
        sub = load(inc, root, _seen + (name,))
        out["chain"] += [c for c in sub["chain"] if c not in out["chain"]]
        out["stage"].update(sub["stage"])
        out["env"].update(sub["env"])
    out["chain"].append(name)
    out["stage"].update(doc.get("stage", {}))
    out["env"].update({k: str(v) for k, v in doc.get("env", {}).items()})
    return out


def override(cfg, sets):
    """Apply ['drafts=0', 'env.FLERP_MOE_GROUPED=0', ...]; the config's name gains '+overrides' so results say so."""
    for s in sets or []:
        k, _, v = s.partition("=")
        if k.startswith("env."):
            cfg["env"][k[4:]] = v
        elif k in STAGE_KEYS:
            t = STAGE_KEYS[k]
            cfg["stage"][k] = (v.lower() in ("1", "true", "on", "yes")) if t is bool else t(v)
        else:
            raise SystemExit(f"--set {k}: stage keys are {sorted(STAGE_KEYS)}, or env.NAME")
    if sets:
        cfg["name"] = cfg["name"] + "+" + ",".join(sets)
    return cfg


def drafts(cfg):
    return int(cfg["stage"].get("drafts", 0)) if cfg["stage"].get("draft_head", False) else 0


def context(cfg, default):
    """Positions of the one seq and of every stage's arena: the config's stage `context`, else the launcher's default."""
    return int(cfg["stage"].get("context", default))


def reuse_tail(cfg):
    """Rank 0's executor key reuse_tail (slot-resident prefix reuse, ORNITH-PCACHE): 0 off."""
    return int(cfg["stage"].get("reuse_tail", 0))


def reuse_grid(cfg):
    return int(cfg["stage"].get("reuse_grid", 0))


def env_prefix(cfg):
    """Shell words `NAME=VALUE ...` for a unit's environment."""
    return " ".join(f"{k}={v}" for k, v in sorted(cfg["env"].items()))
