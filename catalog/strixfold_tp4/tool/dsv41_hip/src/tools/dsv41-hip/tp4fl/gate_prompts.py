#!/usr/bin/env python3
"""gate_prompts.py URL PROMPTS OUT: the TP4-FIRST-LIGHT gate's 8 frozen prompts (716f35d40a), one after another, each a
greedy non-streamed chat request (temperature 0, max_tokens from the file). OUT/replies.jsonl gets one line a prompt
(id, HTTP status, seconds, usage, reply text). The logits SHA of each prompt is not here: rank 0 logs it
({"prompt_logits": ..., "sha256": ..., "rows": ...}, LOGITS_SHA=1) and the runner takes those lines from rank0.log in
order. Exit 0 only when all 8 answered 200."""
import json, sys, time, urllib.request

def main():
    url, prompts, out = sys.argv[1:4]
    doc = json.load(open(prompts))
    model = json.load(urllib.request.urlopen(url + "/v1/models", timeout=30))["data"][0]["id"]
    bad = 0
    with open(out + "/replies.jsonl", "w") as f:
        for p in doc["prompts"]:
            body = {"model": model, "messages": p["messages"], "temperature": doc["temperature"],
                    "max_tokens": doc["max_tokens"], "stream": False}
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.time(); rec = {"id": p["id"]}
            try:
                r = json.load(urllib.request.urlopen(req, timeout=600))
                rec.update(status=200, usage=r.get("usage"), text=r["choices"][0]["message"].get("content"))
            except Exception as e:  # recorded, the rest still run
                rec.update(status=getattr(e, "code", 0), error=str(e)[:300]); bad += 1
            rec["seconds"] = round(time.time() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(json.dumps({k: rec.get(k) for k in ("id", "status", "seconds", "usage")}))
    sys.exit(1 if bad else 0)

main()
