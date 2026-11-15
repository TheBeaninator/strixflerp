#!/usr/bin/env python3
"""The WINJOB argument files for the Ornith 397B window engine (card 83a59b0e07; skill /data/pod/skills/winjob).

    win_args.py kit                       the kit_make arguments (code + libs + nodectl.sh, no weights)
    win_args.py run [--minutes 90] [--set drafts=0 ...]   the win_run arguments: nodes F D A C, the kit, `cmd` = engine.py up
Pipe to a file and pass it to `pod-winjob kit_make|win_run --args FILE` (pod-hands writes the same shape). win_run makes the
window: it places the kit, waits in line for F D A C (never B), takes them, runs cmd on control-node, fetches the rank logs, stops
the user units, closes the window; the keepers bring the StrixFold pair and strixite back."""
import argparse
import json
import sys

CARD = "83a59b0e07"
PLAN = "/data/pod/plans/ornith-engine"
ENGINE = "/data/projects/strixflerp-cfg/tools/ornith-engine/engine.py"   # becomes /data/projects/strixflerp/... once card/83a59b0e07 merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["kit", "run"])
    ap.add_argument("--minutes", type=int, default=90, help="serve this long at most; the window is minutes + 25 (load, cleanup)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="engine.py --set (old behaviour: drafts=0, reply=full, env.FLERP_MOE_GROUPED=0)")
    a = ap.parse_args()
    if a.what == "kit":
        doc = {"_card": CARD, "name": "ornith397",
               "files": {"bin": f"{PLAN}/kit/bin", "lib": f"{PLAN}/kit/lib", "manifests": f"{PLAN}/kit/manifests",
                         "generation_config.json": f"{PLAN}/kit/generation_config.json"}}
    else:
        sets = "".join(f" --set {s}" for s in a.set)
        win = a.minutes + 25
        doc = {"_card": CARD, "nodes": "F D A C", "kit": "ornith397", "for": f"{win}m", "wait_s": "3600",
               "why": "Ornith-1.5 397B window engine ornith397_flerp (card 83a59b0e07)",
               "cmd": f"python3 {ENGINE} up --max-min {a.minutes}{sets}",
               "fetch": [f"{x}:flerp-ornith-run/*" for x in "FDAC"]}
    json.dump(doc, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
