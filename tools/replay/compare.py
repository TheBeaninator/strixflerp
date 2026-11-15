#!/usr/bin/env python3
"""compare.py SELFID_DIR SUMMARY_OUT : compare run A vs run B (one at a time) and the concurrent pairs vs run A seq."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import classes, primary, norm

w, out = sys.argv[1], sys.argv[2]
A = json.load(open(os.path.join(w, "run-A.json"))); B = json.load(open(os.path.join(w, "run-B.json")))
ids = [os.path.basename(p.strip())[:-5] for p in open(os.path.join(w, "list.txt")) if p.strip()]
ex = {i: json.load(open(os.path.join(w, "snap", "strixite", i + ".json"))) for i in ids}
cls = {i: primary(classes(ex[i]["request"], json.loads(ex[i].get("response_raw") or "{}"))) for i in ids}
labels = {i: classes(ex[i]["request"], json.loads(ex[i].get("response_raw") or "{}")) for i in ids}
rows = collections.OrderedDict(); fails = []
def tally(table, key, a, b, i):
    r = table.setdefault(key, [0, 0, 0, []])
    r[0] += 1
    ta, ba = norm(a["raw"]); tb, bb = norm(b["raw"])
    tok = a["status"] == b["status"] == 200 and ta == tb
    byt = tok and ba == bb
    r[1] += tok; r[2] += byt
    if not byt:
        r[3].append(i)
for i in ids:
    if i in A["seq"] and i in B["seq"]:
        tally(rows, cls[i], A["seq"][i], B["seq"][i], i)
pairs = collections.OrderedDict()
for src in (A, B):
    for i, v in src["pairs"].items():
        tally(pairs, cls[i] + f" (run {src['run']})", src["seq"][i], v, i)
L = ["# qwen38-flash-next strixite self-replay (FLERP-SELFID bab6510b88)", "",
     f"Origin: `strix_server.v070tg` (20937dc + d19945e), fleet argv, on control-node's GPU via tools/gpu-queue.sh; two runs A and B, each a fresh process with an empty prompt-cache dir; recorded seed (FreeFitty sends `seed`) kept; max_tokens capped at {A['seq'][ids[0]]['max_tokens'] if ids else '-'}; load {A.get('load_secs')} s / {B.get('load_secs')} s.",
     "Token-identical = same choices[].message (content, reasoning, tool_calls), finish_reason and usage.completion_tokens; byte-identical = whole body equal after dropping id, created, timings, system_fingerprint and each tool call's random `id` (call_<random>, drawn per reply). In the two-in-flight table the alone run came first, so the in-flight request hits the prompt cache and `usage.prompt_tokens_details.cached_tokens` differs: byte-identity there is expected to fail, and token identity is the measure.", "",
     "## One conversation at a time: run A vs run B", "", "| class | requests | token-identical | byte-identical | first differing id |", "|---|---|---|---|---|"]
for k, r in rows.items():
    L.append(f"| {k} | {r[0]} | {r[1]} | {r[2]} | {r[3][0] if r[3] else '-'} |")
L += ["", "## Two conversations in flight at once vs the same request alone (same run)", "",
      "| class (run) | requests | token-identical | byte-identical | first differing id |", "|---|---|---|---|---|"]
for k, r in pairs.items():
    L.append(f"| {k} | {r[0]} | {r[1]} | {r[2]} | {r[3][0] if r[3] else '-'} |")
L += ["", "## Per request", "", "| id | labels | prompt_tokens (recorded) | A secs | B secs | A==B token | A==B bytes |", "|---|---|---|---|---|---|---|"]
for i in ids:
    if i in A["seq"] and i in B["seq"]:
        a, b = A["seq"][i], B["seq"][i]; ta, ba = norm(a["raw"]); tb, bb = norm(b["raw"])
        pt = (json.loads(ex[i].get("response_raw") or "{}").get("usage") or {}).get("prompt_tokens")
        L.append(f"| {i} | {','.join(labels[i])} | {pt} | {a['secs']} | {b['secs']} | {ta == tb} | {ba == bb} |")
L += ["", "Pair (StrixFold) rows: empty. StrixFold against itself needs the pair's GPUs (F+D), which serve FreeFitty, and no pair window was agreed with its owner. The pair half of corpus-v1 exists (151 chats) for when one is."]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
