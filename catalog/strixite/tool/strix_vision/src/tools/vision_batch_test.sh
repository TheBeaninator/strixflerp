#!/usr/bin/env bash
# vision_batch_test.sh - live-engine regression test for strixite issue #29 (card 7e85feb549).
#
# WHAT IT PROVES: an image-carrying conversation (whose session state holds mRoPE position triples) can
# prefill and decode BESIDE text conversations without failing. Pre-fix (upstream issue #29, and the pod
# v050 line before the #30 port), a GpuArbiter verify_batch packed the mRoPE conversation's MTP verifies
# with another slot's run into one scalar-positions forward; the runtime check (runtime/qwen4exp.cpp,
# forward_batch: "pending3.empty() && !history_ever_interleaved && rope_delta == 0") threw and took EVERY
# run of the batch down - the image turn AND its concurrent sibling failed together (seen on B 2026-11-06
# 20:41Z beside a 105,928-token lane turn). Post-fix the arbiter asks LmBackend::batchable(seq) and the
# mRoPE conversation verifies solo, so both conversations succeed.
#
# THE FIXTURE IS GENERATED AT RUNTIME AND SELF-CHECKED, never embedded as a hex string: an earlier copy of
# this test (strixite-v050 fix-29arbiter 8fd40b9/c44bb1f and the spike probe.sh) shipped a base64 8x8 PNG
# whose IHDR/IDAT CRCs were wrong - the engine 400'd it ("decode_image ... stb failed to decode the png") and
# any run that counted itself green on it was a FALSE GREEN (card 7e85feb549 comment #37060). png() below
# builds a real 64x64 solid-blue PNG; assert_png() walks the chunks, re-checks every CRC, inflates the IDAT
# stream and (if PIL is importable) decodes it. A wrong fixture now fails the test locally, not the batch.
#
# Four overlapping rounds, all requests must answer 200:
#   1. a LONG text turn (LONG_WORDS filler codes, default 30000 - measured 176,709 prompt tokens on C
#      2026-11-06 23:1xZ) starts prefilling; an image turn arrives beside it, then both decode; the image
#      answer must name blue,
#   2. the SAME image conversation continues (history: image + its answer + a follow-up, so its mRoPE state is
#      resumed from the prompt cache) beside a fresh short text turn - the landmine round,
#   3. the image conversation again, beside a second long text turn,
#   4. optional (--long-prefill N): the FIELD SHAPE from the card's next_action - a text turn that prefills
#      about N tokens in one slot while the image conversation turns in the other; the REAL size is read back
#      from the response usage (usage.prompt_tokens), so the log states tokens actually prefilled.
# With --require-solo (patched binaries publish the counter) /metrics strix_verify_solo_mrope_total must rise -
# the solo packs prove the gate engaged, not just that nothing broke.
#
# usage: [LONG_WORDS=30000] vision_batch_test.sh http://HOST:PORT [--require-solo] [--long-prefill N]
# exit 0 + "VISION-BATCH-PASS ..." on success; exit 1 + "VISION-BATCH-FAIL ..." otherwise.
# Needs a RUNNING engine with total-slots >= 2, the vision tower loaded, and room for LONG_WORDS
# (~5.9 prompt tokens per filler word; boot1 evidence: LONG_WORDS=30000 -> prompt_tokens=176,709).
set -u
URL=${1:?usage: vision_batch_test.sh http://HOST:PORT [--require-solo] [--long-prefill N]}
shift || true
REQUIRE_SOLO=0
LONG_PREFILL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --require-solo) REQUIRE_SOLO=1 ;;
        --long-prefill) LONG_PREFILL=${2:?--long-prefill needs a token count}; shift ;;
        --long-prefill=*) LONG_PREFILL=${1#--long-prefill=} ;;
        *) echo "VISION-BATCH-FAIL unknown argument: $1"; exit 1 ;;
    esac
    shift
done
case "$LONG_PREFILL" in (''|*[!0-9]*) echo "VISION-BATCH-FAIL --long-prefill needs a number"; exit 1;; esac
LONG_WORDS=${LONG_WORDS:-30000}

solo_mrope() { curl -s -m 5 "$URL/metrics" | awk '/^strix_verify_solo_mrope_total /{print $2}' | head -1; }
M0=$(solo_mrope); M0=${M0:-}

OUT=$(python3 - "$URL" "$LONG_WORDS" "$LONG_PREFILL" <<'EOF'
import base64, hashlib, json, os, struct, sys, threading, time, urllib.request, zlib
url = sys.argv[1]; long_words = int(sys.argv[2]); long_prefill = int(sys.argv[3] or 0)
# fixture colour is overridable so a rerun can dodge the engine-side vision preprocess cache and force a
# FRESH decode in-window; keep IMG_COLOR in step with the colour the answer must name.
FIX_RGB = tuple(int(x) for x in os.environ.get("IMG_RGB", "20, 40, 220").replace(" ", "").split(","))
IMG_COLOR = os.environ.get("IMG_COLOR", "blue")

def png(rgb, n=64):
    raw = b"".join(b"\x00" + bytes(rgb) * n for _ in range(n))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def assert_png(b):
    """Strict local decode: signature, every chunk CRC, IHDR dims, IDAT inflate size, first pixel; then PIL."""
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature"
    i, idat, dims = 8, b"", None
    while True:
        ln = struct.unpack(">I", b[i:i + 4])[0]; typ = b[i + 4:i + 8]; data = b[i + 8:i + 8 + ln]
        crc = struct.unpack(">I", b[i + 8 + ln:i + 12 + ln])[0]
        assert crc == zlib.crc32(typ + data) & 0xffffffff, "CRC mismatch in chunk %r" % typ
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert depth == 8 and ctype == 2, "not 8-bit RGB"
            dims = (w, h)
        elif typ == b"IDAT":
            idat += data
        elif typ == b"IEND":
            break
        i += 12 + ln
    raw = zlib.decompress(idat)
    assert len(raw) == dims[1] * (1 + dims[0] * 3), "inflated IDAT size %d != %d" % (len(raw), dims[1] * (1 + dims[0] * 3))
    assert raw[1:4] == bytes(FIX_RGB), "first pixel is not the fixture colour"
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(b)); im.verify()
        im = Image.open(io.BytesIO(b))
        assert im.size == dims and im.mode == "RGB", "PIL decode mismatch"
    except ImportError:
        pass
    return dims

BLOB = png(FIX_RGB)
DIMS = assert_png(BLOB)
IMG = "data:image/png;base64," + base64.b64encode(BLOB).decode()
print("FIXTURE %dx%d solid RGB%s bytes=%d sha256=%s" % (DIMS[0], DIMS[1], FIX_RGB, len(BLOB),
      hashlib.sha256(BLOB).hexdigest()), flush=True)

fails, results, usage = [], {}, {}
def call(body, tag, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        c = r["choices"][0]["message"]["content"] or ""
        results[tag] = c
        usage[tag] = r.get("usage") or {}
        print("%-8s ok %5.1fs prompt=%s (%d chars) %r" % (tag, time.time() - t0, usage[tag].get("prompt_tokens"), len(c), c[:60]), flush=True)
    except Exception as e:
        body_txt = getattr(e, "read", lambda: b"")()[:300]
        fails.append("%s: %s %s" % (tag, e, body_txt)); print("%-8s FAIL %s %s" % (tag, e, body_txt), flush=True)
def text_body(n, words, k):
    filler = " ".join("w%d" % ((i * 7919 + k) % 100003) for i in range(words))
    return {"model": "qwen3.8-flash-next", "max_tokens": n, "messages": [{"role": "user", "content":
            "Here is a list of codes: " + filler + "\nWrite a short list of colours, one per line, ten lines."}]}
conv = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": IMG}},
         {"type": "text", "text": "What colour is this image? Then describe it in about twenty words."}]}]
prev_tag = ["img-1"]
def img_turn(tag, mt, follow=None):
    global conv
    if follow:
        conv = conv + [{"role": "assistant", "content": results.get(prev_tag[0], IMG_COLOR.capitalize())}, {"role": "user", "content": follow}]
    call({"model": "qwen3.8-flash-next", "max_tokens": mt, "messages": conv}, tag); prev_tag[0] = tag
# Round 1: a LONG text prefill starts first; the image turn arrives while it prefills, then both decode.
t = threading.Thread(target=call, args=(text_body(160, long_words, 1), "long-1")); t.start(); time.sleep(3)
img_turn("img-1", 96); t.join()
# Round 2 (the landmine): the SAME conversation (history = image + answer + follow-up, mRoPE state resumed from the
# prompt cache) beside a fresh text turn.
t = threading.Thread(target=call, args=(text_body(192, 2000, 2), "text-2")); t.start(); time.sleep(1)
img_turn("img-2", 96, "Now say the first word of your answer again and count to five."); t.join()
# Round 3: another overlap of the image conversation against a second long text turn.
t = threading.Thread(target=call, args=(text_body(128, long_words // 2, 3), "long-3")); t.start(); time.sleep(3)
img_turn("img-3", 64, "Name one object that usually has that colour."); t.join()
# Round 4 (optional, the field shape from the card's next_action): ~N-token prefill beside the image
# conversation; the real size is read back from usage below.
if long_prefill:
    words = max(2000, int(long_prefill / 5.9))
    t = threading.Thread(target=call, args=(text_body(16, words, 4), "long-4")); t.start(); time.sleep(10)
    img_turn("img-4", 64, "Reply with just the colour name."); t.join()
    got = (usage.get("long-4") or {}).get("prompt_tokens")
    print("LONG-PREFILL asked=%d prompt_tokens=%s" % (long_prefill, got), flush=True)
if IMG_COLOR not in results.get("img-1", "").lower(): fails.append("img-1 did not name the colour %s" % IMG_COLOR)
if fails: print("EXCEPTIONS " + " | ".join(fails)); sys.exit(1)
print("ALL-REQUESTS-SUCCEEDED"); sys.exit(0)
EOF
)
RC=$?
echo "$OUT"

if [ $RC -ne 0 ]; then
    echo "VISION-BATCH-FAIL concurrent image+text turns raised, the image answer was wrong, or the fixture failed its local decode: see lines above"
    exit 1
fi

M1=$(solo_mrope); M1=${M1:-}
DELTA=""
if [ -n "$M0" ] && [ -n "$M1" ]; then
    DELTA=$(awk -v a="$M0" -v b="$M1" 'BEGIN{print b-a}')
    echo "strix_verify_solo_mrope_total: $M0 -> $M1 (delta $DELTA)"
    if [ "$REQUIRE_SOLO" = 1 ] && ! awk -v d="$DELTA" 'BEGIN{exit !(d>0)}'; then
        echo "VISION-BATCH-FAIL the batchable() gate never solo-packed the mRoPE conversation (delta $DELTA)"
        exit 1
    fi
elif [ "$REQUIRE_SOLO" = 1 ]; then
    echo "VISION-BATCH-FAIL --require-solo but the engine publishes no strix_verify_solo_mrope_total (pre-fix binary)"
    exit 1
fi
echo "VISION-BATCH-PASS image conversation + text conversations prefilled and decoded together, zero batch throws"
exit 0
