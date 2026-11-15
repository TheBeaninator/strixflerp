#!/usr/bin/env python3
"""Prefill from a cache on the live DSV4.1 pair: resume a kept prefix, append K new tokens, record what the server
reports. Ordinary greedy requests, no hold. Usage: prefill_dsv41_warm_ladder.py <out.json> [--prefix file.json] [--serve-url URL] [--serving-conf F] | --plan"""
import json, os, random, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_nodes  # noqa: E402

URL = None           # the serve URL, from the live serving conf (pair_nodes) or --serve-url; set in main()
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name river "
         "table window garden letter music paper morning forest silver copper bridge harbor engine ladder candle").split()
LADDER = [250, 1000, 2000, 4000, 8000, 16000, 32000]


def get(path):
    return json.load(urllib.request.urlopen(URL + path, timeout=10))


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    return out, time.time() - t


def filler(k, seed):
    r = random.Random(seed)
    return "Notes, unrelated to anything above: " + " ".join(r.choice(WORDS) for _ in range(k)) + \
        "\nReply with the single word: ok"


def wait_idle():
    for _ in range(60):
        live = get("/health").get("live", {})
        if live.get("connections", 1) == 0 and live.get("waiting", 1) == 0:
            return True
        time.sleep(2)
    return False


def ask(base, k, seed, rows, arm):
    body = dict(base)
    body["messages"] = base["messages"] + [{"role": "user", "content": filler(k, seed)}]
    body.update(stream=False, max_tokens=2, temperature=0)
    idle = wait_idle()
    out, wall = post(body)
    u, tf = out.get("usage", {}), out.get("tensorfold", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    row = {"arm": arm, "k_words": k, "seed": seed, "idle_before": idle, "prompt_tokens": u.get("prompt_tokens"),
           "cached_tokens": cached, "new_tokens": (u.get("prompt_tokens") or 0) - (cached or 0),
           "prefill_seconds": tf.get("prefill_seconds"), "wall_s": round(wall, 3), "at": time.strftime("%H:%M:%SZ", time.gmtime())}
    if row["prefill_seconds"] and row["new_tokens"]:
        row["new_tok_per_s"] = round(row["new_tokens"] / row["prefill_seconds"], 1)
    rows.append(row)
    print(json.dumps(row), flush=True)


def main():
    global URL
    over = sys.argv[sys.argv.index("--serve-url") + 1] if "--serve-url" in sys.argv else None
    conf = sys.argv[sys.argv.index("--serving-conf") + 1] if "--serving-conf" in sys.argv else pair_nodes.LIVE_CONF
    pair = pair_nodes.load(conf, serve_url=over)
    URL = pair["serve_url"]
    print(pair_nodes.describe(pair), file=sys.stderr, flush=True)
    if "--plan" in sys.argv:
        return
    out_path = sys.argv[1]
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else None
    rows = []
    if prefix:
        src = json.load(open(prefix))
        base = {k: src[k] for k in ("model", "shared_messages", "messages", "tools") if k in src}
        for rep in (1, 2):
            for k in LADDER:
                ask(base, k, 1000 * rep + k, rows, "standard-cache")
    # small cache: a prefix of our own, nothing shared (nothing learned to disk); the first request keeps it in RAM
    r = random.Random(7)
    small = {"model": "dsv41", "messages": [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Background text: " + " ".join(r.choice(WORDS) for _ in range(4300))},
        {"role": "assistant", "content": "Understood."}]}
    ask(small, 250, 5, rows, "small-cache-seed")
    for rep in (1, 2):
        for k in LADDER:
            ask(small, k, 2000 * rep + k, rows, "small-cache")
    json.dump({"url": URL, "pair": pair, "health_after": get("/health"), "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
