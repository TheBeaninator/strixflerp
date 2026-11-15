#!/usr/bin/env python3
"""drive2.py ACTIVE_THREADS... : keep at most 4 librarian-dsv41 threads busy; start the next prompt only when one is done
(last message assistant_text, nothing pending). Stops when prompts run out or STOP file exists."""
import json, os, subprocess, sys, time, tomllib, urllib.request
D = "/data/pod/flerp-replay/pair-drive"
c = tomllib.load(open("/data/pod/freefitty/freefitty.toml", "rb"))
H = {"Authorization": "Bearer " + c["operator_token"]}
FF = "/home/pod/.local/bin/freefitty"


def log(*a):
    with open(f"{D}/drive.log", "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + " ".join(map(str, a)) + "\n")


def threads():
    r = urllib.request.Request("http://127.0.0.1:8191/threads", headers=H)
    return {t["thread"]: t for t in json.load(urllib.request.urlopen(r, timeout=60))}


prompts = json.load(open(f"{D}/prompts.json"))
nxt = int(open(f"{D}/next").read()) if os.path.exists(f"{D}/next") else 0
active = list(sys.argv[1:])
while True:
    if os.path.exists(f"{D}/STOP"):
        log("STOP file"); break
    try:
        by = threads()
    except Exception as e:
        log("threads-err", e); time.sleep(30); continue
    for t in list(active):
        s = by.get(t, {})
        if s.get("last") == "assistant_text" and not s.get("pending"):
            log("done", t, s.get("messages")); active.remove(t)
    while len(active) < 4 and nxt < len(prompts):
        out = subprocess.run([FF, "-c", "/data/pod/freefitty/freefitty.toml", "say", "-e", "librarian-dsv41", prompts[nxt]],
                             capture_output=True, text=True)
        try:
            th = json.loads(out.stdout)["thread"]; active.append(th); log("say", nxt, th)
        except Exception:
            log("say-fail", nxt, out.stdout[:200], out.stderr[:200])
        nxt += 1; open(f"{D}/next", "w").write(str(nxt))
    if not active and nxt >= len(prompts):
        log("all done"); break
    time.sleep(20)
