#!/usr/bin/env python3
"""vision_check.py - image-input correctness + cache-xfer smoke against ONE strixite server (card 7e85feb549).

usage: vision_check.py http://HOST:PORT OUT.jsonl [--images DIR] [--no-xfer]

1. solid colours: 64x64 PNGs generated here (red, green, blue, yellow, white, black); "what colour" must name it.
2. real images from DIR (default /data/pod/plans/strixite-vision/images): screen_1 (keys strixite, AGPL),
   graph_12 (keys 12 nodes, yes), one photo described freely (printed for a human read, no key).
3. cache-xfer: a ~1.5k-token text turn, POST /cache/export of the same body (200 + bytes), POST /cache/import of
   those bytes back (200), an empty-body export answers 400 (route on), then the turn again reports cached tokens.
Every request: temperature 0, thinking off. Exit 0 iff every keyed check passes.
"""
import base64, json, os, struct, sys, time, urllib.error, urllib.request, zlib

URL = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
IMG = "/data/pod/plans/strixite-vision/images"
if "--images" in sys.argv: IMG = sys.argv[sys.argv.index("--images") + 1]
XFER = "--no-xfer" not in sys.argv
MODEL = "qwen3.8-flash-next"

def png(rgb, n=64):
    raw = b"".join(b"\x00" + bytes(rgb) * n for _ in range(n))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def post(path, body, timeout=600, raw=False):
    data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/octet-stream" if raw else "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()

def ask(img_bytes, mime, q, mt=96):
    url = "data:%s;base64,%s" % (mime, base64.b64encode(img_bytes).decode())
    body = {"model": MODEL, "max_tokens": mt, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]}]}
    t0 = time.time(); st, b = post("/v1/chat/completions", body)
    if st != 200: return {"http": st, "error": b[:300].decode("utf-8", "replace")}
    r = json.loads(b); u = r.get("usage", {})
    return {"http": st, "answer": (r["choices"][0]["message"].get("content") or "").strip(),
            "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"), "wall_s": round(time.time() - t0, 2)}

rows, fails = [], []
def rec(name, r, keys=None):
    r["name"] = name
    if keys is not None:
        low = r.get("answer", "").lower(); hits = [k for k in keys if k.lower() in low]
        r["keys"], r["hits"] = keys, hits
        if len(hits) != len(keys): fails.append(name)
    elif r.get("http") != r.pop("expect", 200): fails.append(name)
    rows.append(r); print(json.dumps(r), flush=True)

for colour, rgb in [("red", (220, 20, 20)), ("green", (20, 180, 40)), ("blue", (20, 40, 220)),
                    ("yellow", (240, 220, 20)), ("white", (255, 255, 255)), ("black", (0, 0, 0))]:
    rec("solid_" + colour, ask(png(rgb), "image/png", "What single colour fills this image? Answer with one word."), [colour])

def f(p): return open(os.path.join(IMG, p), "rb").read()
rec("screen_1", ask(f("screens/screen_1.png"), "image/png", "What is the repository name and the license shown on this page? Answer briefly."), ["strixite", "AGPL"])
rec("graph_12", ask(f("diagrams/graphs/graph_12.png"), "image/png", "How many nodes does this graph have? Is there an edge between B and C? Answer briefly."), ["12", "yes"])
photos = sorted(os.listdir(os.path.join(IMG, "photos")))
if photos: rec("photo_" + photos[0], ask(f("photos/" + photos[0]), "image/jpeg", "Describe this photo in one sentence.", 80))

if XFER:
    st, b = post("/cache/export", b"", 30, raw=True)
    rec("xfer_empty_export_400", {"http": st, "expect": 400, "body": b[:120].decode("utf-8", "replace")}, None)
    filler = " ".join("item-%d is filed under shelf %d." % (i, i * 7 % 97) for i in range(220))
    msgs = [{"role": "user", "content": "Here is an inventory list.\n" + filler + "\nWhich shelf holds item-42? Answer with the number only."}]
    body = {"model": MODEL, "messages": msgs, "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    st, b = post("/v1/chat/completions", body); r1 = json.loads(b) if st == 200 else {}
    rec("xfer_turn1", {"http": st, "answer": (r1.get("choices") or [{}])[0].get("message", {}).get("content", ""), "usage": r1.get("usage")}, None)
    for _ in range(10):  # the turn's cache entry is written after the answer goes out
        time.sleep(2); st, blob = post("/cache/export", body, 120)
        if st == 200: break
    rec("xfer_export", {"http": st, "bytes": len(blob)}, None)
    if st == 200 and len(blob) > 0:
        st2, b2 = post("/cache/import", blob, 120, raw=True)
        rec("xfer_import", {"http": st2, "body": b2[:200].decode("utf-8", "replace")}, None)
    st, b = post("/v1/chat/completions", body); r2 = json.loads(b) if st == 200 else {}
    rec("xfer_turn2", {"http": st, "usage": r2.get("usage")}, None)

with open(OUT, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
print("VISION-CHECK-%s %d rows, fails: %s" % ("FAIL" if fails else "PASS", len(rows), fails or "none"))
sys.exit(1 if fails else 0)
