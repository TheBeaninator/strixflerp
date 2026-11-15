import json, subprocess, re, os
os.environ["KANBAN_DB"] = "/data/pod/kanban.db"
lst = subprocess.run(["akb", "list", "--column", "todo"], capture_output=True, text=True).stdout
ids = re.findall(r"\b([0-9a-f]{10})\b", lst)
out = []
for i in ids:
    t = subprocess.run(["akb", "read", i], capture_output=True, text=True).stdout
    m = re.search(r"^title: (.*)$", t, re.M); g = re.search(r"^goal: \|\n((?:  .*\n)+)", t, re.M)
    w = re.search(r"^WHY: (.*)$", t, re.M)
    if not (m and w) or "<SCRUB" in t: continue
    out.append(f"Reading list for this pod task (what the ark holds that I should read before starting it):\n\nTask: {m.group(1)}\nWhy: {w.group(1)}\n" + (g.group(1) if g else ""))
    if len(out) >= 14: break
json.dump(out, open("/data/pod/flerp-replay/pair-drive/prompts.json", "w"))
print(len(out)); [print(repr(t[100:220])) for t in out]
