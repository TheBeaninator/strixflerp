#!/usr/bin/env python3
"""Module gate: one module compiles on its own for its arch with its toolchain.

    ./flerp gate module <module-id>

Generates build/gate-module/<id>/CMakeLists.txt holding only that module (include paths from its requirements,
each interface resolved to its one provider), configures and builds it, and writes
gate/results/modules/<id>.json. A header-only module is compiled through a generated file that includes each of
its headers. Run on the build host; this proves compilation only, not parity.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flerplib import assemble  # noqa: E402
from flerplib.catalog import Catalog  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cat = Catalog(ROOT)
    m = cat.by_id.get(sys.argv[1])
    if m is None:
        sys.exit(f"no module {sys.argv[1]}")
    tree = os.path.join(ROOT, "build", "gate-module", m.id)
    res_dir = os.path.join(ROOT, "gate", "results", "modules")
    os.makedirs(res_dir, exist_ok=True)
    out = {"module": m.id, "host": os.uname().nodename, "arch": m.build.get("arch"),
           "toolchain": m.build.get("toolchain"), "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    errors = assemble.generate_module(cat, m, tree)
    log = ""
    if not errors:
        for cmd in (["cmake", "-G", "Ninja", "-S", tree, "-B", os.path.join(tree, "ninja")],
                    ["ninja", "-C", os.path.join(tree, "ninja")]):
            p = subprocess.run(cmd, capture_output=True, text=True)
            log += f"$ {' '.join(cmd)}\n{p.stdout}{p.stderr}"
            if p.returncode != 0:
                errors.append(f"{cmd[0]} failed with {p.returncode}")
                break
    out.update({"pass": not errors, "errors": errors, "log_tail": log.splitlines()[-15:]})
    with open(os.path.join(res_dir, m.id + ".json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"{m.id}: {'PASS' if not errors else 'FAIL ' + '; '.join(errors)}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
