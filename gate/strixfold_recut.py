#!/usr/bin/env python3
"""Re-cut catalog/strixfold to another commit of the pod's StrixFold fork and its deployment. Run on the laptop:

    python3 gate/strixfold_recut.py <commit> <deploy-dir-name> [--rank1 IP]   e.g.  f64b5ff f64b5ff2e28b

The serving build on the pair's workers keeps moving; the recipe gate compares like with like only when the catalogue,
the kit pins and the gate's reference are the commit that is deployed. This does all of it, reading only:
  1. on control-node (tools/pod-run.sh): `git archive <commit>` of the fork into build/strixfold-<commit>; the
     deployed kit manifest's sha256 and DEPLOY.json from ~/tfp/deploy/<deploy-dir-name> on rank 1's worker (read-only
     ssh; the live serving conf's rank 1 by gate/pair_nodes.py, or --rank1 IP);
     the build tree that made that kit (the worktree whose zig-out manifest has that sha256) and, from its Zig cache,
     the directory name each of the deployed HIP code objects was compiled into (the kit output pins);
  2. on the laptop: the export pulled to build/ (tools/pod-pull.sh); docs/harvest/harvest_strixfold.py re-run on
     it with the new commit (verbatim copy, manifests, ORIGIN.toml); the pins and the reference written into
     recipes/dsv41-flash.strixfold-tp2{,-abi}.toml, gate/dsv41-flash.strixfold-tp2.toml and the gate's base conf.
It then prints the commands that rebuild and re-run the check, origin and build stages. It does not touch the pair,
commit, or change anything on the pod outside /data/projects/strixflerp/build.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK = "/data/pod/plans/strixfold/fork"
# where the deployment is read from: rank 1 of the live serving pair, resolved on control-node by gate/pair_nodes.py
# from /data/pod/plans/strixfold/tp2-128k.conf, unless `--rank1 IP` names a worker

POD = r'''
import glob, hashlib, json, os, subprocess, sys
commit, deploy, fork, worker = sys.argv[1:5]
sys.path.insert(0, "gate")
import pair_nodes
pair = None if worker else pair_nodes.load()
worker = worker or pair["rank1"]["mgmt"]
run = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
full = run(f"git -C {fork} rev-parse {commit}").stdout.strip()
short = run(f"git -C {fork} rev-parse --short=7 {commit}").stdout.strip()
out = {"commit": short, "commit_full": full, "pair": pair, "worker": worker}
exp = f"build/strixfold-{short}"
r = run(f"rm -rf {exp} && mkdir -p {exp} && git -C {fork} archive {full} | tar -x -C {exp}")
out["export"], out["export_rc"] = exp, r.returncode
ssh = f"ssh -o BatchMode=yes -o ConnectTimeout=8 pod@{worker}"
d = f"~/tfp/deploy/{deploy}"
out["deployed_commit"] = run(f"{ssh} 'python3 -c \"import json,os;print(json.load(open(os.path.expanduser(\\\"{d}/DEPLOY.json\\\")))[\\\"commit\\\"])\"'").stdout.strip()
kit = "native-hip/share/tensorfold/hip/gfx1151/dsv41/manifest.json"
out["kit_manifest_sha256"] = run(f"{ssh} 'sha256sum {d}/{kit}'").stdout.split(" ")[0]
trees = [l.split()[0] for l in run(f"git -C {fork} worktree list").stdout.splitlines() if l.strip()]
out["build_tree"], out["pins"] = None, {}
for t in trees:
    m = os.path.join(t, "zig-out", kit)
    if not os.path.isfile(m) or hashlib.sha256(open(m, "rb").read()).hexdigest() != out["kit_manifest_sha256"]:
        continue
    out["build_tree"] = t
    by = {o["origin"]: o["sha256"] for o in json.load(open(m))["objects"] if o["origin"].endswith(".hip")}
    for p in sorted(glob.glob(os.path.join(t, ".zig-cache/o/*/*.hsaco"))):
        name = os.path.basename(p)
        if by.get(name[:-6] + ".hip") == hashlib.sha256(open(p, "rb").read()).hexdigest():
            out["pins"][name] = p.split(os.sep)[-2]
    out["hip_objects_in_kit"] = len(by)
    break
print("RECUT " + json.dumps(out))
'''


def sub(path, pairs):
    with open(path) as f:
        text = f.read()
    for pattern, repl, flags in pairs:
        text, n = re.subn(pattern, repl, text, flags=flags)
        if n == 0:
            sys.exit(f"{path}: nothing matches {pattern!r}")
    with open(path, "w") as f:
        f.write(text)


def main():
    args = sys.argv[1:]
    over = ""
    if "--rank1" in args:
        i = args.index("--rank1")
        over = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    commit, deploy = args
    script = f"cat > build/strixfold_recut_pod.py <<'PYEOF'\n{POD}\nPYEOF\npython3 build/strixfold_recut_pod.py {commit} {deploy} {FORK} '{over}'"
    r = subprocess.run([os.path.join(ROOT, "tools/pod-run.sh"), script], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RECUT ")), None)
    if line is None:
        sys.exit("control-node did not answer:\n" + r.stdout[-800:] + r.stderr[-800:])
    info = json.loads(line[6:])
    WORKER = info["worker"]
    print(json.dumps(info, indent=1))
    if info["export_rc"] != 0 or not info["commit"]:
        sys.exit(f"{commit} is not a commit of {FORK}")
    if info["deployed_commit"] != info["commit_full"]:
        sys.exit(f"~/tfp/deploy/{deploy} on {WORKER} is commit {info['deployed_commit'] or '(unreadable)'}, not {info['commit_full']}")
    if not info["build_tree"] or len(info["pins"]) != info.get("hip_objects_in_kit"):
        sys.exit("no worktree of the fork holds the build of the deployed kit (its zig-out manifest and Zig cache): the "
                 "pins cannot be read; the recipe would build a kit that differs in the compile unit ids")
    short = info["commit"]
    subprocess.check_call([os.path.join(ROOT, "tools/pod-pull.sh"), info["export"]], cwd=ROOT)

    # the harvest, on the export
    sub(os.path.join(ROOT, "docs/harvest/harvest_strixfold.py"),
        [(r"(O = Origin\('strixfold', 'strixfold', 'https://github.com/Shoestring-dev/Strixfold', ')[0-9a-f]+'", rf"\g<1>{short}'", 0),
         (r'(?m)^commit    = "[0-9a-f]+"', f'commit    = "{short}"', 0)])
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "up"))
        os.makedirs(os.path.join(tmp, "sup"))
        os.symlink(os.path.join(ROOT, info["export"]), os.path.join(tmp, "up", "strixfold"))
        env = dict(os.environ, STRIXFLERP_UPSTREAM=os.path.join(tmp, "up"), STRIXFLERP_SUPPLEMENT=os.path.join(tmp, "sup"))
        subprocess.check_call([sys.executable, os.path.join(ROOT, "docs/harvest/harvest_strixfold.py")], env=env, cwd=ROOT)

    # the pins and the reference
    width = max(len(n) for n in info["pins"]) + 2
    pins = "".join(f'{chr(34) + n + chr(34):<{width}} = "{d}"\n' for n, d in sorted(info["pins"].items()))
    block = (f"# The cache directory names the reference build (control-node:{info['build_tree']} at {short}, the build the\n"
             f"# deployment {deploy} on the serving pair's workers was copied from) compiled each code object into, read from its Zig cache by\n"
             f"# gate/strixfold_recut.py. With them the kit comes out byte for byte: manifest sha256\n"
             f"# {info['kit_manifest_sha256']}.\n"
             f"# Without this table the build is the origin's own and its eleven HIP objects differ in the compile unit id only.\n"
             f"[build.hipcc_output_pins]\n{pins}\n")
    for name in ("recipes/dsv41-flash.strixfold-tp2.toml", "recipes/dsv41-flash.strixfold-tp2-abi.toml"):
        sub(os.path.join(ROOT, name),
            [(r"# The cache directory names.*?(?=\[build\.installs\])", lambda m: block, re.S),
             (r'deployment = "[^"]*"', f'deployment = "~/tfp/deploy/{deploy} on the serving pair workers"', 0)])
    sub(os.path.join(ROOT, "recipes/dsv41-flash.strixfold-tp2.toml"),
        [(r"# StrixFold [0-9a-f]+ \(the pod's fork\)", f"# StrixFold {short} (the pod's fork)", 0)])
    sub(os.path.join(ROOT, "gate/dsv41-flash.strixfold-tp2.toml"),
        [(r'reference_kit_manifest_sha256 = "[0-9a-f]+"', f'reference_kit_manifest_sha256 = "{info["kit_manifest_sha256"]}"', 0),
         (r"# The kit of the reference deployment.*", f"# The kit of the reference deployment (~/tfp/deploy/{deploy} on the serving pair workers, built in {info['build_tree']})", 0),
         (r'reference_deploy   = "\$HOME/tfp/deploy/[0-9a-f]+"', f'reference_deploy   = "$HOME/tfp/deploy/{deploy}"', 0),
         (r'reference_commit   = "[0-9a-f]+"', f'reference_commit   = "{info["commit_full"]}"', 0)])
    sub(os.path.join(ROOT, "gate/dsv41-flash.strixfold-tp2.conf"),
        [(r"DEPLOY_DIR='\$HOME/tfp/deploy/[0-9a-f]+'", f"DEPLOY_DIR='$HOME/tfp/deploy/{deploy}'", 0)])
    subprocess.check_call([sys.executable, os.path.join(ROOT, "flerp"), "check", "--origin", "strixfold"], cwd=ROOT)
    print(f"\nre-cut to {short} (deployment {deploy}, kit {info['kit_manifest_sha256'][:16]}...). Next:\n"
          "  tools/pod-sync.sh\n"
          "  tools/pod-run.sh './flerp gate recipe dsv41-flash.strixfold-tp2 --stages check,origin,build'\n"
          "  tools/pod-run.sh './flerp assemble dsv41-flash.strixfold-tp2-abi && build/dsv41-flash.strixfold-tp2-abi/build.sh'\n"
          "  tools/pod-run.sh './flerp gate recipe dsv41-flash.strixfold-tp2 --stages pair --plan'\n"
          "then review `git status catalog/strixfold recipes gate` and commit those paths.")


if __name__ == "__main__":
    main()
