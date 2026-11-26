# serve_lib.sh: sourced by serve-rank0.sh, serve-rank1.sh and first_light.sh. Resolves serve.conf (or $TF_SERVE_CONF)
# for one rank on this box into SERVE_CMD (the argv) and SERVE_ENV (the whole environment the rank starts with:
# nothing else leaks in), checks the inputs, and writes the resolved form with identities (resolved.json).
# shellcheck shell=bash

SERVE_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVE_CONF=${TF_SERVE_CONF:-$SERVE_TOOLS/serve.conf}

die() { echo "$(basename "$0"): $*" >&2; exit 1; }

# serve_transport: the collectives' wire. tcp (COLL_KIND host or proxy) or rdma (rdma_proxy.zig, always kind rdma: one
# QP a pair carries every gather that fits a slot, so there is no second proxy thread). A gather larger than a slot
# goes through tcp_host under either transport (rdma_proxy.zig gatherBegin), so TP_STRIPE, COLL_STRIPE_MIB and
# COLL_WIRE_THREADS shape those gathers under rdma as they do under tcp. rccl: comm.zig's Comm over RCCL (TF_DS_COLL=rccl,
# TP4-RCCL-COLL 01574d76b3), the only wire of the four-node split (WORLD=4): tcp_coll, tcp_proxy and rdma_proxy are
# two-rank; COLL_KIND resolves to rccl and the COLL_* proxy settings do not apply.
serve_transport() {
  TRANSPORT=${TRANSPORT:-tcp} RDMA_DEV0=${RDMA_DEV0:-} RDMA_DEV1=${RDMA_DEV1:-} RDMA_IFNAME=${RDMA_IFNAME:-} RDMA_GID=${RDMA_GID:-3}
  RCCL_LIB=${RCCL_LIB:-} RAIL_IFNAME=${RAIL_IFNAME:-}
  case $TRANSPORT in
    tcp) ;;
    rccl)
      COLL_KIND=rccl
      [[ $RDMA_GID =~ ^[0-9]+$ ]] && [ "$RDMA_GID" -le 255 ] || die "RDMA_GID must be a GID index 0..255, not $RDMA_GID";;
    rdma)
      COLL_KIND=rdma
      [ "${COLL_PROXY_STRIPE_KIB:-0}" = 0 ] || die "TRANSPORT=rdma: COLL_PROXY_STRIPE_KIB must be empty or 0 (no second proxy thread)"
      if [ -n "$RDMA_DEV0$RDMA_DEV1" ]; then
        [ -n "$RDMA_DEV0" ] && [ -n "$RDMA_DEV1" ] || die "TRANSPORT=rdma: RDMA_DEV0 and RDMA_DEV1 come together (both or neither)"
      fi
      [[ $RDMA_GID =~ ^[0-9]+$ ]] && [ "$RDMA_GID" -le 255 ] || die "RDMA_GID must be a GID index 0..255, not $RDMA_GID";;
    *) die "TRANSPORT must be tcp, rdma or rccl, not $TRANSPORT";;
  esac
}

# serve_nodes (WORLD=4): the four boxes, rank g = position g in NODES (tp4.conf). NODES, NODE_IPS (mgmt) and NODE_RAILS
# (rail) have WORLD entries each, the letters distinct, and rank 0's rail address is MASTER_IP (it listens there: rank g
# connects to TP_PORT + g - 1, fan.zig). Sets NODE, NODE_IP and NODE_RAIL for RANK, and RANK_HOST_<g> (the ssh name
# first_light.sh uses from the driver box: RANK<g>_HOST when the conf sets it, else pod@<mgmt address>).
serve_nodes() {
  local -a n i r; read -ra n <<< "${NODES:-}"; read -ra i <<< "${NODE_IPS:-}"; read -ra r <<< "${NODE_RAILS:-}"
  [ "${#n[@]}" = "$WORLD" ] && [ "${#i[@]}" = "$WORLD" ] && [ "${#r[@]}" = "$WORLD" ] ||
    die "WORLD=$WORLD needs NODES, NODE_IPS and NODE_RAILS with $WORLD entries each (have ${#n[@]}, ${#i[@]}, ${#r[@]})"
  [ "$(printf '%s\n' "${n[@]}" | sort -u | wc -l)" = "$WORLD" ] || die "NODES '${NODES}' names a box twice"
  local g h
  for g in "${!n[@]}"; do
    [[ ${n[g]} =~ ^[A-Z]$ ]] || die "NODES: '${n[g]}' is not a node letter"
    [[ ${i[g]} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [[ ${r[g]} =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
      die "NODE_IPS / NODE_RAILS: rank $g's addresses '${i[g]}' / '${r[g]}' are not IPv4 addresses"
    h=RANK${g}_HOST; printf -v "RANK_HOST_$g" '%s' "${!h:-pod@${i[g]}}"
  done
  [ "${r[0]}" = "$MASTER_IP" ] || die "MASTER_IP $MASTER_IP is not rank 0's rail address ${r[0]} (NODE_RAILS): ranks 1-3 connect there"
  NODE=${n[RANK]} NODE_IP=${i[RANK]} NODE_RAIL=${r[RANK]}
}

# serve_load RANK: the config for RANK on this box (paths expanded with this box's $HOME, RANK, and at WORLD=4 HALF
# (the TP2 rank r of node g = 2p + r: g % 2, whose Engram half the node holds) and PAIR (p = g / 2)).
serve_load() {
  RANK=$1
  [ -f "$SERVE_CONF" ] || die "no config $SERVE_CONF"
  # shellcheck source=serve.conf
  source "$SERVE_CONF"
  case ${WORLD:-} in 2|4) ;; *) die "WORLD must be 2 (the TP2 pair) or 4 (the four-node 2D split), not ${WORLD:-unset}";; esac
  [[ $RANK =~ ^[0-9]$ ]] && [ "$RANK" -lt "$WORLD" ] || die "rank $RANK is not a rank of WORLD=$WORLD"
  HALF=$((RANK % 2)) PAIR=$((RANK / 2))
  NODE=${NODE:-} NODE_IP=${NODE_IP:-} NODE_RAIL=${NODE_RAIL:-}
  [ "$WORLD" = 4 ] && serve_nodes
  DEPLOY_DIR_RAW=$DEPLOY_DIR  # unexpanded: another box's own $HOME (first_light.sh names rank 0's tools with it)
  local v
  # in this order: a path may name one expanded before it ($BUNDLE)
  for v in DEPLOY_DIR BUNDLE MODEL_DIR RANK_CACHE DS_KIT ENGRAM_DIR TOKEN_MAP RUN_ROOT; do eval "$v=\"${!v}\""; done
  PROMPT_DISK_GIB=${PROMPT_DISK_GIB:-0}
  [ -n "${PROMPT_DISK_DIR:-}" ] && eval "PROMPT_DISK_DIR=\"$PROMPT_DISK_DIR\""
  # the image settings serve.conf describes and TEST_FAULT (TF_DS_TEST_FAULT, recovery tests); empty when not set
  VISION_DIR=${VISION_DIR:-} VISION_BIAS_SHA256=${VISION_BIAS_SHA256:-} TEST_FAULT=${TEST_FAULT:-} MAX_IMAGES=${MAX_IMAGES:-}
  IMAGE_TOKENS=${IMAGE_TOKENS:-} ENCODE_CACHE_MIB=${ENCODE_CACHE_MIB:-}
  [ -z "$MAX_IMAGES" ] || [[ $MAX_IMAGES =~ ^[1-9][0-9]*$ ]] || die "MAX_IMAGES must be an image count from 1 (empty: the server's 32), not $MAX_IMAGES"
  [ -z "$IMAGE_TOKENS" ] || { [[ $IMAGE_TOKENS =~ ^[1-9][0-9]{0,5}$ ]] && [ "$IMAGE_TOKENS" -le 65536 ]; } ||
    die "IMAGE_TOKENS must be a request's image tokens from 1 to 65536 (empty: the engine's 4096), not $IMAGE_TOKENS"
  [ -z "$ENCODE_CACHE_MIB" ] || { [[ $ENCODE_CACHE_MIB =~ ^(0|[1-9][0-9]{0,4})$ ]] && [ "$ENCODE_CACHE_MIB" -le 65536 ]; } ||
    die "ENCODE_CACHE_MIB must be the picture cache's MiB from 0 (off) to 65536 (empty: 512), not $ENCODE_CACHE_MIB"
  [ -n "$VISION_DIR" ] && eval "VISION_DIR=\"$VISION_DIR\""
  EDIT=${EDIT:-} EDIT_SHA256=${EDIT_SHA256:-}
  [ -n "$EDIT" ] && eval "EDIT=\"$EDIT\""
  MASK_EXPERTS=${MASK_EXPERTS:-}
  [ -n "$MASK_EXPERTS" ] && eval "MASK_EXPERTS=\"$MASK_EXPERTS\""
  case $MODALITIES in text|text_image) ;; *) die "MODALITIES must be text or text_image, not $MODALITIES";; esac
  BIN=$DEPLOY_DIR/native-hip/bin
  KIT=$DEPLOY_DIR/$KIT_SUBDIR
  [ "$PARALLEL" -ge 1 ] && [ "$PARALLEL" -le "$STREAMS" ] || die "PARALLEL $PARALLEL must be 1 .. STREAMS ($STREAMS)"
  local f size rows=
  # (SERVE_PLAN=1, first_light.sh plan on another box: this box's files are not read; the rank's own check does it)
  [ "${SERVE_PLAN:-0}" = 1 ] && rows=$((CONTEXT + 49))
  [ "${SERVE_PLAN:-0}" = 1 ] || for f in rope-plain-cos rope-plain-sin rope-compressed-cos rope-compressed-sin; do
    size=$(stat -L -c %s "$DS_KIT/$f.f32" 2>/dev/null) || die "no RoPE table $DS_KIT/$f.f32 (DS_KIT)"
    [ -z "$rows" ] || [ $((size / 128)) -lt "$rows" ] && rows=$((size / 128))
  done
  # the engine reads a stream's most positions (CONTEXT), a round's 48 rows and one (model.zig rope_rows), 128 B a row
  [ "$CONTEXT" -le $((rows - 49)) ] || die "CONTEXT $CONTEXT is past the RoPE tables' $rows rows (at most $((rows - 49)))"
  # POOL: the streams' shared pool (unset: CONTEXT, one stream may take it all); above CONTEXT, several long streams
  # share it while each request stays within CONTEXT (rank 0 TF_DS_POOL, rank 1 --pool; both ranks --context CONTEXT)
  POOL=${POOL:-$CONTEXT}
  [[ $POOL =~ ^[0-9]+$ ]] && [ "$POOL" -ge "$CONTEXT" ] || die "POOL $POOL must be a position count of at least CONTEXT ($CONTEXT)"
  serve_transport
  [ "$WORLD" = 2 ] || [ "$TRANSPORT" = rccl ] || [ "$TRANSPORT" = rdma ] || die "WORLD=$WORLD needs TRANSPORT=rccl or rdma: tcp_coll and tcp_proxy are two-rank (rdma_proxy serves 4 since TP4-RDMA-N f782ee5476)"
  case $COLL_KIND in host|proxy|rdma|rccl) ;; *) die "COLL_KIND must be host or proxy (over tcp), not $COLL_KIND";; esac
  [ "$GRAPHS" = 0 ] || [ "$COLL_KIND" != host ] || die "GRAPHS=1 needs COLL_KIND=proxy, TRANSPORT=rdma or TRANSPORT=rccl (tcp_host is not capturable)"
  if [ "$GRAPHS" != 0 ] && [ "$COLL_KIND" != host ] && [ "$COLL_KIND" != rccl ]; then
    if [ "$WORLD" = 4 ]; then
      # world 4 (rdma_proxy RdmaProxyN, TP4-QUICKWINS 034bf2c6cd): a slot carries 4 blocks of at most the slot less four
      # 64-byte trailers, split 4 ways, 64-byte aligned (rdma_core Layout.gatherChunk). The largest decode gather is the
      # head's: this node's hw columns (253 of the 505 128-column blocks of the 64,640-wide vocab half = 32,384) x 4 B a
      # row. A larger block still captures, but every graph replay then runs it as serialized slot chunks: refused here,
      # with the COLL_SLOT_KIB that fits it.
      local hb=$((ROUND_ROWS * 32384 * 4)) sb=$(( (COLL_SLOT_KIB * 1024 + 4095) / 4096 * 4096 )) ch need
      ch=$(( (sb - 4 * 64) / 4 / 64 * 64 )); need=$(( (4 * hb + 4 * 64 + 4095) / 4096 * 4 ))
      [ "$hb" -le "$ch" ] || die "GRAPHS=1 at WORLD=4: the head gather at ROUND_ROWS=$ROUND_ROWS is 4 blocks of $hb B, a COLL_SLOT_KIB=$COLL_SLOT_KIB slot carries blocks of $ch B, so every graph would run it as $(( (hb + ch - 1) / ch )) serialized slot chunks; set COLL_SLOT_KIB >= $need (rdma_proxy's world-4 default is 8704)"
    else
      [ $((ROUND_ROWS * 64640 * 4)) -le $((COLL_SLOT_KIB * 1024)) ] ||
        die "GRAPHS=1: the head gather at ROUND_ROWS=$ROUND_ROWS ($((ROUND_ROWS * 64640 * 4)) B) exceeds COLL_SLOT_KIB=$COLL_SLOT_KIB"
    fi
  fi
  # COLL_SWAP12=1 (TF_COLL_SWAP12, every rank alike: the handshake compares it): one-row quarters gathered straight into
  # place (rdma_core Word.gatherSwap12, prompt2d quarters); rdma_proxy's per-peer path only
  [ "${COLL_SWAP12:-0}" = 0 ] || { [ "$WORLD" = 4 ] && [ "$COLL_KIND" = rdma ]; } || die "COLL_SWAP12=1 needs WORLD=4 and TRANSPORT=rdma (rdma_proxy's four-rank gathers)"
  # (a configuration written before these settings existed runs with them off)
  SPAWN_UNIT=${SPAWN_UNIT:-0}
  case $SPAWN_UNIT in 0|1) ;; *) die "SPAWN_UNIT must be 0 or 1, not $SPAWN_UNIT";; esac
  REPLAY_FLOOR=${REPLAY_FLOOR:-0} PLACE_VICTIM=${PLACE_VICTIM:-0} HC_SIDE=${HC_SIDE:-0} ENGRAM_AIO=${ENGRAM_AIO:-0} TOP_K_MAX=${TOP_K_MAX:-0} SEED_RANDOM=${SEED_RANDOM:-0}
  L2_PREFETCH=${L2_PREFETCH:-0} LOOP_GUARD=${LOOP_GUARD:-1} DRAFT_TEMPERATURE=${DRAFT_TEMPERATURE:-0} GPU_COUNTS=${GPU_COUNTS:-0}
  [[ $DRAFT_TEMPERATURE =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v t="$DRAFT_TEMPERATURE" 'BEGIN { exit !(t <= 100) }' \
    || die "DRAFT_TEMPERATURE must be a number from 0 (off) to 100, not $DRAFT_TEMPERATURE"
  DRAFT_BYPASS_AT=${DRAFT_BYPASS_AT:-0}
  [[ $DRAFT_BYPASS_AT =~ ^(0|[2-8])$ ]] || die "DRAFT_BYPASS_AT must be 0 or 2..8"
  FILL_SLICE_MS=${FILL_SLICE_MS:-0}
  [[ $FILL_SLICE_MS =~ ^[0-9]{1,6}$ ]] || die "FILL_SLICE_MS must be a whole number of ms up to 999999 (0: whole prompt passes)"
  [[ ${FINISH_LEFT:-} =~ ^[0-9]{0,6}$ ]] || die "FINISH_LEFT must be empty or a whole number of tokens up to 999999 (0: no stream is waited for)"
  [[ ${QUEUE_MAX:-} =~ ^([0-9]{1,6}|off)?$ ]] || die "QUEUE_MAX must be empty, a whole number of requests up to 999999, or off"
  [[ ${QUEUE_MIB:-} =~ ^([0-9]{1,7}|off)?$ ]] || die "QUEUE_MIB must be empty, a whole number of MiB up to 9999999, or off"
  [[ ${QUEUE_RETRY_S:-} =~ ^[0-9]{0,4}$ ]] || die "QUEUE_RETRY_S must be empty or a whole number of seconds up to 9999"
  # the switch acts only on drafted TP2 shared rounds that can hold that many streams; elsewhere it is off
  if [ "$DRAFTS" != 1 ] || [ "$WORLD" != 2 ] || [ "$DRAFT_BYPASS_AT" -gt "$STREAMS" ]; then DRAFT_BYPASS_AT=0; fi
  for v in REPLAY_FLOOR PLACE_VICTIM GPU_COUNTS HC_SIDE ENGRAM_AIO SEED_RANDOM L2_PREFETCH LOOP_GUARD; do case ${!v} in 0|1) ;; *) die "$v must be 0 or 1, not ${!v}";; esac; done
  serve_loop_policy
  [[ $TOP_K_MAX =~ ^[0-9]+$ ]] || die "TOP_K_MAX must be a whole number (0: off), not $TOP_K_MAX"
  # sf-disk (off unless PROMPT_DISK_GIB > 0): it writes what the memory tier keeps, so that must be on
  [[ $PROMPT_DISK_GIB =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "PROMPT_DISK_GIB must be a number of GiB (0: off), not $PROMPT_DISK_GIB"
  # the disk tier runs at WORLD 2 and 4 (TP4-DISK-STATES 602dfeb13b, e33be4f4: each rank keeps its own key.r<g>);
  # learned states (LEARN_DIR) stay TP2-only: at another WORLD they stay off with one line
  if serve_disk_on && [ "$WORLD" != 2 ] && [ "$WORLD" != 4 ]; then
    echo "serve_lib: rank $RANK: PROMPT_DISK_GIB=$PROMPT_DISK_GIB off at WORLD=$WORLD (the disk tier runs at WORLD 2 and 4)" >&2
    PROMPT_DISK_GIB=0
  fi
  if [ -n "${LEARN_DIR:-}" ] && [ "$WORLD" != 2 ]; then
    echo "serve_lib: rank $RANK: LEARN_DIR=$LEARN_DIR off at WORLD=$WORLD (learned prompt states are TP2-only until TP4-DISK-STATES)" >&2
    LEARN_DIR= LEARN_GIB=
  fi
  if serve_disk_on; then
    awk -v g="${PROMPT_CACHE_GIB:-0}" 'BEGIN { exit !(g > 0) }' || die "PROMPT_DISK_GIB=$PROMPT_DISK_GIB needs PROMPT_CACHE_GIB > 0 (the disk tier writes the kept states)"
    local k
    for k in PROMPT_DISK_FLOOR_GIB PROMPT_DISK_QUEUE_MIB PROMPT_DISK_MIN_TOKENS PROMPT_DISK_RATE_MIB; do
      [ -z "${!k:-}" ] || [[ ${!k} =~ ^[0-9]+$ ]] || die "$k must be a whole number, not ${!k}"
    done
  fi
}

# serve_disk_on: true when the configuration turns sf-disk on (PROMPT_DISK_GIB > 0).
# serve_loop_policy: LOOP_POLICY (pod card f64f606edd) = off | close | stop | count; unset or empty: LOOP_GUARD's old
# meaning (1 or unset = close, 0 = off), so a configuration written before LOOP_POLICY keeps its behaviour.
serve_loop_policy() {
  if [ -z "${LOOP_POLICY:-}" ]; then
    if [ "${LOOP_GUARD:-1}" = 1 ]; then LOOP_POLICY=close; else LOOP_POLICY=off; fi
  fi
  case $LOOP_POLICY in off|close|stop|count) ;; *) die "LOOP_POLICY must be off, close, stop or count, not $LOOP_POLICY";; esac
}

serve_disk_on() { awk -v g="${PROMPT_DISK_GIB:-0}" 'BEGIN { exit !(g > 0) }'; }

# serve_check: every input this rank reads is here (paths only; verify_inputs.sh checks the bytes).
serve_check() {
  local f
  for f in "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" "$MODEL_DIR/chat_template.jinja" "$RANK_CACHE" "$DS_KIT/engram.json" "$ENGRAM_DIR" "$TOKEN_MAP" "$KIT/manifest.json" \
           "$BIN/tensorfold-native" "$BIN/tf-dsv41-lanes" "$BUNDLE/inputs_manifest.json" "$DEPLOY_DIR/DEPLOY.json"; do
    [ -e "$f" ] || die "rank $RANK: missing $f"
  done
  local sum
  if [ "$MODALITIES" = text_image ]; then
    local bias=$DS_KIT/vision/gate_bias_vl.safetensors
    [ -f "$bias" ] || die "rank $RANK: MODALITIES=text_image needs $bias (tools/dsv41-zig/kit/make_bias_vl.py; deploy_rank0.sh --ds-kit copies it)"
    [ -n "$VISION_BIAS_SHA256" ] || die "rank $RANK: MODALITIES=text_image needs VISION_BIAS_SHA256 (the routing biases' sha256)"
    sum=$(sha256sum "$bias" | cut -d' ' -f1)
    [ "$sum" = "$VISION_BIAS_SHA256" ] || die "rank $RANK: $bias is $sum, the config pins $VISION_BIAS_SHA256"
    if [ "$RANK" = 0 ]; then
      [ -n "$VISION_DIR" ] && [ -f "$VISION_DIR/SHA256SUMS" ] || die "rank 0: MODALITIES=text_image needs VISION_DIR, the tower's sidecar with its SHA256SUMS (tools/dsv41-zig/kit/make_vision_sidecar.py); now '$VISION_DIR'"
      (cd "$VISION_DIR" && sha256sum --quiet -c SHA256SUMS) >&2 || die "rank 0: the vision sidecar $VISION_DIR does not match its SHA256SUMS"
    fi
  fi
  if [ -n "$EDIT" ]; then
    [ -f "$EDIT" ] || die "rank $RANK: missing the edit $EDIT (tools/dsv41-hip/make_edit.py)"
    [ -n "$EDIT_SHA256" ] || die "rank $RANK: EDIT needs EDIT_SHA256 (the edit file's sha256, which make_edit.py prints)"
    sum=$(sha256sum "$EDIT" | cut -d' ' -f1)
    [ "$sum" = "$EDIT_SHA256" ] || die "rank $RANK: $EDIT is $sum, the config pins $EDIT_SHA256"
    # the pack the edit was made from (its metadata's base, REPO@COMMIT) is the bundle's checkpoint
    python3 - "$EDIT" "$BUNDLE/inputs_manifest.json" <<'PY' || die "rank $RANK: $EDIT was made for another pack than $BUNDLE's"
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    meta = json.loads(f.read(struct.unpack("<Q", f.read(8))[0])).get("__metadata__") or {}
dc = json.load(open(sys.argv[2])).get("declared_checkpoint") or {}
repo, _, rev = str(meta.get("base", "")).partition("@")
sys.exit(0 if (repo, rev) == (dc.get("repository"), dc.get("revision")) else 1)
PY
  fi
  # MASK_EXPERTS (pod card 5bb471d3b9): the same mask file on every rank's box (the engine refuses a peer whose mask
  # differs: its digest is in the handshake's plan text); the engine parses it and refuses a bad line itself
  if [ -n "$MASK_EXPERTS" ]; then
    [ -f "$MASK_EXPERTS" ] || die "rank $RANK: missing the expert mask $MASK_EXPERTS (MASK_EXPERTS: 'layer:expert' lines, route_join.py --mask-out)"
  fi
  sum=$(sha256sum "$BUNDLE/inputs_manifest.json" | cut -d' ' -f1)
  [ "$sum" = "$BUNDLE_SHA256" ] || die "rank $RANK: inputs_manifest.json is $sum, the config pins $BUNDLE_SHA256"
  # U16: RANK_CACHE holds exactly the cache file the manifest names for this rank (the engine takes any
  # rank<R>of<W>-*.bin, so a stale or substituted cache under another key is refused here), and with VERIFY_REPORT=1
  # this box's verify_inputs.sh report (BUNDLE.verify-rank<R>-<host>.json) passed for this manifest
  python3 - "$BUNDLE/inputs_manifest.json" "$RANK_CACHE" "$RANK" "$WORLD" "${VERIFY_REPORT:-1}" \
    "$BUNDLE.verify-rank$RANK-$(hostname).json" "$BUNDLE_SHA256" <<'PY' || die "rank $RANK: the inputs are not the verified bundle's"
import json, os, sys
manifest, cache, rank, world, need_report, report, sha = sys.argv[1:8]
want = [c["build"]["report"]["file"] for c in json.load(open(manifest))["rank_caches"] if c["rank"] == int(rank)]
have = sorted(n for n in os.listdir(cache) if n.startswith(f"rank{rank}of{world}-") and n.endswith(".bin"))
if len(want) != 1 or have != want:
    sys.exit(f"rank cache {cache} holds {have or 'no cache'}, the manifest names {want}")
if need_report == "1":
    if not os.path.exists(report):
        sys.exit(f"no verify report {report} (tools/dsv41-hip/verify_inputs.sh)")
    r = json.load(open(report))
    if r.get("status") != "pass" or r.get("rank") != int(rank) or r.get("manifest_sha256") != sha:
        sys.exit(f"verify report {report}: status {r.get('status')}, rank {r.get('rank')}, manifest "
                 f"{r.get('manifest_sha256', 'not recorded')}; wanted pass for rank {rank} and manifest {sha}")
PY
  [ "$TRANSPORT" = rdma ] && serve_rdma_check
  [ "$TRANSPORT" = rccl ] && serve_rccl_check
  return 0
}

# serve_rdma_check (TRANSPORT=rdma): this rank's device, port, GID, memlock and uverbs (rdma_check.py). The slab the
# proxy registers is 2 x 8 slots x COLL_SLOT_KIB (the engine's TF_COLL_SLOTS default); memlock must hold at least
# that, and 256 MiB.
serve_rdma_check() {
  local dv=RDMA_DEV$RANK need args=()
  need=$(( (2 * 8 * COLL_SLOT_KIB + 1023) / 1024 ))
  [ "$need" -ge 256 ] || need=256
  [ -n "${!dv:-}" ] && args+=(--dev "${!dv}")
  [ -n "$RDMA_IFNAME" ] && args+=(--ifname "$RDMA_IFNAME")
  python3 "$SERVE_TOOLS/rdma_check.py" "${args[@]}" --gid "$RDMA_GID" --memlock-mib "$need" >&2 ||
    die "rank $RANK: the rdma check failed on $(hostname) (rdma_check.py above)"
}

# serve_rail_ifname / serve_rail_hca (TRANSPORT=rccl): the rail's interface (RAIL_IFNAME, else the one holding this
# rank's rail address: NODE_RAIL at WORLD=4, MASTER_IP for rank 0 of a pair) and its RDMA device (RDMA_DEV<rank>, else
# the box's only ibdev), for NCCL_SOCKET_IFNAME and NCCL_IB_HCA. Under SERVE_PLAN=1 (another box) they are named, not read.
serve_rail_ifname() {
  local a=${NODE_RAIL:-$MASTER_IP}
  [ -n "$RAIL_IFNAME" ] && { echo "$RAIL_IFNAME"; return; }
  [ "${SERVE_PLAN:-0}" = 1 ] && { echo "<iface-of-$a>"; return; }
  ip -4 -o addr show | awk -v a="$a" '{ split($4, p, "/"); if (p[1] == a) { print $2; exit } }'
}
serve_rail_hca() {
  local dv=RDMA_DEV$RANK d
  [ -n "${!dv:-}" ] && { echo "${!dv}"; return; }
  [ "${SERVE_PLAN:-0}" = 1 ] && { echo "<only-ibdev-on-${NODE:-rank$RANK}>"; return; }
  d=$(ls /sys/class/infiniband 2>/dev/null)
  [ "$(wc -w <<< "$d")" = 1 ] && echo "$d"
}

# serve_rccl_check (TRANSPORT=rccl): the library, the rail interface and device resolve on this box, and memlock is
# unlimited (RCCL registers its buffers: ibv_reg_mr fails under a limit, the rail recipe's LimitMEMLOCK=infinity).
serve_rccl_check() {
  [ -z "$RCCL_LIB" ] || [ -f "$RCCL_LIB" ] || die "rank $RANK: RCCL_LIB $RCCL_LIB is not on $(hostname)"
  [ -n "$(serve_rail_ifname)" ] || die "rank $RANK: no interface holds the rail address ${NODE_RAIL:-$MASTER_IP} on $(hostname) (RAIL_IFNAME)"
  [ -n "$(serve_rail_hca)" ] || die "rank $RANK: no single RDMA device on $(hostname) (RDMA_DEV$RANK names one)"
  [ "$(ulimit -l)" = unlimited ] || die "rank $RANK: memlock is $(ulimit -l) KiB on $(hostname); RCCL needs unlimited (LimitMEMLOCK=infinity)"
}

# serve_idle (U25): this box is idle before a rank starts: no other engine process, and MemAvailable at least the
# rank's peak (PEAK_GIB) plus 8 GiB.
serve_idle() {
  local others avail
  others=$(pgrep -x -d, 'ds4|ds4-server|ds4-bench|tensorfold-nati|tf-dsv41-lanes|tf-dsv41-layer' || true)
  [ -z "$others" ] || die "rank $RANK: $(hostname) is not idle: engine process(es) $others"
  avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
  [ "$avail" -ge $((PEAK_GIB + 8)) ] || die "rank $RANK: MemAvailable ${avail} GiB on $(hostname), a rank needs ${PEAK_GIB} + 8 GiB"
}

# serve_env: SERVE_ENV, the rank's whole environment (env -i starts it with exactly this).
serve_env() {
  SERVE_ENV=(HOME="$HOME" USER="${USER:-}" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8
    TF_GPU_STRICT="$GPU_STRICT" TF_GPU_COUNTS="$GPU_COUNTS" TF_HIP_DEVICE="$HIP_DEVICE" TENSORFOLD_HIP_MEMORY_LIMIT_GB="$MEMORY_LIMIT_GB"
    TF_DS_MODALITIES="$MODALITIES" TF_DS_GRAPHS="$GRAPHS" TF_DS_ARENA_GIB="$ARENA_GIB"
    TF_COLL_STRIPE_MIB="$COLL_STRIPE_MIB" TF_COLL_RTO_MIN_US="$COLL_RTO_MIN_US" TF_COLL_PROGRESS_MS="$COLL_PROGRESS_MS"
    TF_COLL_STAGING="$COLL_STAGING" TF_COLL_KIND="$COLL_KIND" ${AUX:+"TF_DS_AUX=$AUX"} TF_DS_REPLAY_FLOOR="$REPLAY_FLOOR"
    TF_DS_PLACE_VICTIM="$PLACE_VICTIM")
  # (TF_DS_AUX only when AUX is set: unset = hip in tri_aux.Which.fromEnv, and an empty value is BadSetting at load, 716f35d40a)
  # the proxy settings serve both slot collectives (tcp_proxy and rdma_proxy); the stripe thread is tcp_proxy's only
  case $COLL_KIND in proxy|rdma) SERVE_ENV+=(TF_COLL_PROXY_CPU="$COLL_PROXY_CPU" TF_COLL_SLOT_KIB="$COLL_SLOT_KIB");; esac
  [ "${COLL_SWAP12:-0}" = 1 ] && SERVE_ENV+=(TF_COLL_SWAP12=1)
  # CLOCK_MARKS (TF_DS_CLOCK_MARKS, every rank): with tf-dsv41-lanes --profile, GPU clocks at the round phase marks (lanes_main.zig)
  [ -n "${CLOCK_MARKS:-}" ] && SERVE_ENV+=(TF_DS_CLOCK_MARKS="$CLOCK_MARKS")
  [ "$COLL_KIND" = proxy ] && SERVE_ENV+=(TF_COLL_PROXY_STRIPE_KIB="${COLL_PROXY_STRIPE_KIB:-0}" TF_COLL_PROXY_CPU2="${COLL_PROXY_CPU2:-7}")
  # RCCL (TP4-RCCL-COLL's rail recipe): the rail interface and device, RoCE v2 GID, GPU-direct via dma-buf
  if [ "$TRANSPORT" = rccl ]; then
    SERVE_ENV+=(TF_DS_COLL=rccl NCCL_SOCKET_IFNAME="$(serve_rail_ifname)" NCCL_IB_HCA="$(serve_rail_hca)" NCCL_IB_GID_INDEX="$RDMA_GID"
      NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 NCCL_DMABUF_ENABLE=1 NCCL_DEBUG="${NCCL_DEBUG:-WARN}")
    [ -n "$RCCL_LIB" ] && SERVE_ENV+=(TF_RCCL_LIB="$RCCL_LIB")
  fi
  if [ "$TRANSPORT" = rdma ]; then
    # TF_DS_COLL=tcp: the model takes the transport handed in (tcp_coll.open, which with TF_COLL_KIND=rdma opens
    # rdma_proxy: two ranks, or RdmaProxyN at world 3-4), not RCCL (comm.zig chosen; the world-4 default is RCCL)
    SERVE_ENV+=(TF_DS_COLL=tcp TF_RDMA_GID_INDEX="$RDMA_GID")
    if [ "$WORLD" = 4 ]; then
      # world 4 (TP4-RDMA-N): each rank names its own device, one name (RDMA_DEV<rank>, else the box's only rail
      # device: serve_rail_hca), or without RDMA_DEV<g> its rail interface. A two-name list covers ranks 0 and 1
      # only: rank_nic's "names no device for rank 2", BadSetting (716f35d40a run-2003)
      [ -z "$RDMA_DEV0$RDMA_IFNAME" ] && SERVE_ENV+=(TF_RDMA_IFNAME="$(serve_rail_ifname)")
      [ -n "$RDMA_DEV0" ] && SERVE_ENV+=(TF_RDMA_DEVICES="$(serve_rail_hca)")
    else
      [ -n "$RDMA_DEV0" ] && SERVE_ENV+=(TF_RDMA_DEVICES="$RDMA_DEV0,$RDMA_DEV1")   # rank order: rank 0's, rank 1's
    fi
    [ -n "$RDMA_IFNAME" ] && SERVE_ENV+=(TF_RDMA_IFNAME="$RDMA_IFNAME")
  fi
  [ -n "${COLL_GPU_WAIT_S:-}" ] && SERVE_ENV+=(TF_COLL_GPU_WAIT_S="$COLL_GPU_WAIT_S")  # (a conf without them: the engine's)
  [ -n "${MEMORY_RESERVE_GIB:-}" ] && SERVE_ENV+=(TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB")
  [ -n "${COLL_GPU_WARN_S:-}" ] && SERVE_ENV+=(TF_COLL_GPU_WARN_S="$COLL_GPU_WARN_S")
  # COLL_TIME (1 or sync; unset or empty: off): TF_DS_COLL_TIME, rank 0's per-kind collective time on its lanes lines (coll.Timer)
  [ -n "${COLL_TIME:-}" ] && SERVE_ENV+=(TF_DS_COLL_TIME="$COLL_TIME")
  [ -n "$TP_DATA_PORT" ] && SERVE_ENV+=(TF_TP_DATA_PORT="$TP_DATA_PORT")
  [ -n "$TP_STRIPE" ] && SERVE_ENV+=(TF_TP_STRIPE="$TP_STRIPE")
  if [ -n "${COLL_WIRE_THREADS:-}" ]; then
    case $COLL_WIRE_THREADS in 1|2) ;; *) die "COLL_WIRE_THREADS must be 1 or 2 (empty: the engine's 1), not $COLL_WIRE_THREADS";; esac
    [ "$COLL_WIRE_THREADS" = 1 ] || [ -n "$TP_STRIPE" ] || die "COLL_WIRE_THREADS=2 needs TP_STRIPE (a second data link for the second thread)"
    SERVE_ENV+=(TF_COLL_WIRE_THREADS="$COLL_WIRE_THREADS")
  fi
  [ -n "$ROUND_MS" ] && SERVE_ENV+=(TF_DS_ROUND_MS="$ROUND_MS")
  [ -n "$DRAFT_MS" ] && SERVE_ENV+=(TF_DS_DRAFT_MS="$DRAFT_MS")
  [ -n "${FIX_K:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_FIX_K="$FIX_K")
  [ -n "${FROZEN_TIMING:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_LANES_FROZEN_TIMING="$FROZEN_TIMING")
  [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_DRAFT_BYPASS_AT="$DRAFT_BYPASS_AT")
  [ "$RANK" = 0 ] && SERVE_ENV+=(TF_FILL_SLICE_MS="$FILL_SLICE_MS")
  [ -n "${FINISH_LEFT:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_FINISH_LEFT="$FINISH_LEFT")
  # SF-QUEUE-CAP (51591f7ba8): the server's cap on waiting requests and the bytes they hold, rank 0 only (its HTTP side)
  [ -n "${QUEUE_MAX:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_QUEUE_MAX="$QUEUE_MAX")
  [ -n "${QUEUE_MIB:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_QUEUE_MIB="$QUEUE_MIB")
  [ -n "${QUEUE_RETRY_S:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_QUEUE_RETRY_S="$QUEUE_RETRY_S")
  [ -n "${TRACE_ROUNDS:-}" ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_TRACE_ROUNDS="$TRACE_ROUNDS")
  [ -n "${GRAPHS_EAGER:-}" ] && SERVE_ENV+=(TF_DS_GRAPHS_EAGER="$GRAPHS_EAGER")
  # SPEED-FUSE-1 (b7269694f8): the fused hc chain, read by Round.init on every rank (round.zig), so both ranks get it
  [ -n "${HC_CHAIN:-}" ] && SERVE_ENV+=(TF_HC_CHAIN="$HC_CHAIN")
  # SPEED-FUSE-2 (31e3ab5061): the fused router launch, read by Round.init on every rank
  [ -n "${FUSE2:-}" ] && SERVE_ENV+=(TF_FUSE2="$FUSE2")
  # TP4-ATTN (e5cb84a995): TF_DS_TP4_FILL, read by the weights loader on every rank (a 2D node's halved attention linears)
  [ "${TP4_FILL:-0}" = 0 ] || SERVE_ENV+=(TF_DS_TP4_FILL="$TP4_FILL")
  # TP4-VGATHER (e96f0f5fa0): TF_DS_VGATHER, read when the world-4 collective opens (every rank). 1 = a multi-row
  # equal-width quarters is assembled with two copy_rows16; the all-gather is the one issued with it off.
  [ "${VGATHER:-0}" = 0 ] || SERVE_ENV+=(TF_DS_VGATHER="$VGATHER")
  # TP4-REAP-STATS (023c5d2ad9): passive Edge-REAP counters. Every rank reads the schedule (the handshake's
  # plan token). Rank 0 also gets the snapshot dir, the snapshot interval and the served model id. Off (0 or
  # unset): none of these are exported, and the engine's plan text stays the stock one.
  if [ "${REAP_STATS:-0}" = 1 ]; then
    SERVE_ENV+=(TF_DS_REAP_STATS=1)
    SERVE_ENV+=(TF_DS_REAP_EVERY="${REAP_EVERY:-8}")
    SERVE_ENV+=(TF_DS_REAP_PREFILL_ROWS="${REAP_PREFILL_ROWS:-4}")
    SERVE_ENV+=(TF_DS_REAP_BUCKETS="${REAP_BUCKETS:-16}")
    if [ "$RANK" = 0 ]; then
      [ -n "${REAP_DIR:-}" ] && SERVE_ENV+=(TF_DS_REAP_DIR="$REAP_DIR")
      SERVE_ENV+=(TF_DS_REAP_SNAPSHOT_MIN="${REAP_SNAPSHOT_MIN:-10}")
      [ -n "${SERVE_NAME:-}" ] && SERVE_ENV+=(TF_DS_SERVED_NAME="$SERVE_NAME")
    fi
  fi
  # ROUTE-RECORDS (28153e18f3): per-request routing records, rank 0 only (the server and the record writer live
  # there; no other rank changes and the plan text stays the same). Off (0 or unset): nothing is exported.
  if [ "${ROUTE_RECORDS:-0}" = 1 ] && [ "$RANK" = 0 ]; then
    SERVE_ENV+=(TF_DS_ROUTE_RECORDS=1)
    SERVE_ENV+=(TF_DS_ROUTE_RECORDS_DIR="${ROUTE_RECORDS_DIR:-}")
    SERVE_ENV+=(TF_DS_ROUTE_RECORDS_TOPK="${ROUTE_RECORDS_TOPK:-16}")
    SERVE_ENV+=(TF_DS_ROUTE_RECORDS_MAX_MB="${ROUTE_RECORDS_MAX_MB:-2048}")
    [ -n "${SERVE_NAME:-}" ] && SERVE_ENV+=(TF_DS_SERVED_NAME="$SERVE_NAME")
  fi
  # SERVE-ACTS (4f9ba05838): per-request hidden-state dumps, row injection and logits, rank 0 only (the server, the
  # writer and the switches live there; ranks 1-3 take a dump pass's replay and chunks from rank 0's frames, and the
  # plan text stays the same). Off (ACTS_DIR empty or unset): nothing is exported and a request's acts is refused.
  if [ -n "${ACTS_DIR:-}" ] && [ "$RANK" = 0 ]; then
    SERVE_ENV+=(TF_DS_ACTS_DIR="$ACTS_DIR")
    [ -n "${ACTS_ROWS_DIR:-}" ] && SERVE_ENV+=(TF_DS_ACTS_ROWS_DIR="$ACTS_ROWS_DIR")
    [ -n "${ACTS_REQUEST_MB:-}" ] && SERVE_ENV+=(TF_DS_ACTS_REQUEST_MB="$ACTS_REQUEST_MB")
    [ -n "${ACTS_TOTAL_MB:-}" ] && SERVE_ENV+=(TF_DS_ACTS_TOTAL_MB="$ACTS_TOTAL_MB")
    [ -n "${ACTS_QUEUE_MB:-}" ] && SERVE_ENV+=(TF_DS_ACTS_QUEUE_MB="$ACTS_QUEUE_MB")
    [ -n "${ACTS_DUMP_ROWS:-}" ] && SERVE_ENV+=(TF_DS_ACTS_DUMP_ROWS="$ACTS_DUMP_ROWS")
  fi
  # the prefetch's forked stream overlaps the graph's main branch only with HIP's segment scheduling (a debug variable)
  [ "$L2_PREFETCH" = 1 ] && SERVE_ENV+=(DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1)
  # the split doubles Chunk's cap-independent buffers (+474 MB at 16K, +0.62-0.68 GiB per rank at 63,488): serve.conf keeps it on (PREFILL_SPLIT_LARGE=1); LARGE=0 clears it above 16384
  if [ -n "${PREFILL_SPLIT:-}" ] && [ "$PREFILL_SPLIT" != 0 ] && [ "$CONTEXT" -gt 16384 ] && [ "${PREFILL_SPLIT_LARGE:-0}" != 1 ]; then
    echo "serve_lib: PREFILL_SPLIT off at CONTEXT $CONTEXT (> 16384, arena cost unmeasured; PREFILL_SPLIT_LARGE=1 keeps it)" >&2
    PREFILL_SPLIT=
  fi
  [ -n "${PREFILL_SPLIT:-}" ] && SERVE_ENV+=(TF_DS_PREFILL_SPLIT="$PREFILL_SPLIT")
  [ -n "${PREFILL_SPLIT_ZIP:-}" ] && SERVE_ENV+=(TF_DS_PREFILL_SPLIT_ZIP="$PREFILL_SPLIT_ZIP")
  # both ranks alike (the handshake compares it); never dropped here: without the split the engine refuses it (BadSetting)
  if [ -n "${PREFILL_ZIP2:-}" ] && [ "$PREFILL_ZIP2" != 0 ]; then
    [ -n "${PREFILL_SPLIT:-}" ] || echo "serve_lib: PREFILL_ZIP2=$PREFILL_ZIP2 without PREFILL_SPLIT: the engine will refuse it" >&2
    SERVE_ENV+=(TF_DS_PREFILL_ZIP2="$PREFILL_ZIP2")
  fi
  [ -n "${LOGITS_SHA:-}" ] && SERVE_ENV+=(TF_DS_LOGITS_SHA="$LOGITS_SHA")
  [ -n "${WINDOW_LOGITS:-}" ] && SERVE_ENV+=(TF_DS_WINDOW_LOGITS="$WINDOW_LOGITS")
  # sf-disk, both ranks alike (the handshake compares " disk=1"); nothing at all when off
  if serve_disk_on; then
    mkdir -p "${PROMPT_DISK_DIR:-$HOME/strixfold-cache/prefix}" && chmod 700 "${PROMPT_DISK_DIR:-$HOME/strixfold-cache/prefix}" ||
      die "rank $RANK: cannot make PROMPT_DISK_DIR ${PROMPT_DISK_DIR:-$HOME/strixfold-cache/prefix}"
    SERVE_ENV+=(TF_DS_PROMPT_DISK_GIB="$PROMPT_DISK_GIB" TF_DS_PROMPT_DISK_DIR="${PROMPT_DISK_DIR:-$HOME/strixfold-cache/prefix}")
    [ -n "${PROMPT_DISK_FLOOR_GIB:-}" ] && SERVE_ENV+=(TF_DS_PROMPT_DISK_FLOOR_GIB="$PROMPT_DISK_FLOOR_GIB")
    [ -n "${PROMPT_DISK_QUEUE_MIB:-}" ] && SERVE_ENV+=(TF_DS_PROMPT_DISK_QUEUE_MIB="$PROMPT_DISK_QUEUE_MIB")
    [ -n "${PROMPT_DISK_MIN_TOKENS:-}" ] && SERVE_ENV+=(TF_DS_PROMPT_DISK_MIN_TOKENS="$PROMPT_DISK_MIN_TOKENS")
    [ -n "${PROMPT_DISK_RATE_MIB:-}" ] && SERVE_ENV+=(TF_DS_PROMPT_DISK_RATE_MIB="$PROMPT_DISK_RATE_MIB")
  fi
  [ "$MODALITIES" = text_image ] && [ "$RANK" = 0 ] && SERVE_ENV+=(TF_DS_VISION_DIR="$VISION_DIR")
  # (empty: unset, the engine's 4096; rank 1 has it as --image-tokens, serve_lanes_cmd)
  [ "$MODALITIES" = text_image ] && [ "$RANK" = 0 ] && [ -n "${IMAGE_TOKENS:-}" ] && SERVE_ENV+=(TF_DS_IMAGE_TOKENS="$IMAGE_TOKENS")
  [ "$MODALITIES" = text_image ] && [ "$RANK" = 0 ] && [ -n "${ENCODE_CACHE_MIB:-}" ] && SERVE_ENV+=(TF_DS_ENCODE_CACHE_MIB="$ENCODE_CACHE_MIB")
  # python3 -I (the server's Pillow child, pil.zig) ignores PYTHONPATH: a Pillow shipped in the deploy (pylib/) comes by TF_PIL_PATH
  [ "$MODALITIES" = text_image ] && [ "$RANK" = 0 ] && [ -d "$DEPLOY_DIR/pylib/PIL" ] && SERVE_ENV+=(TF_PIL_PATH="$DEPLOY_DIR/pylib")
  [ -n "$TEST_FAULT" ] && SERVE_ENV+=(TF_DS_TEST_FAULT="$TEST_FAULT")
  [ -n "$EDIT" ] && SERVE_ENV+=(TF_DS_EDIT="$EDIT")
  # every rank alike (empty or unset: the variable is absent, the base launch)
  [ -n "${MASK_EXPERTS:-}" ] && SERVE_ENV+=(TF_DS_MASK_EXPERTS="$MASK_EXPERTS")
  # the launch's identity in the peers' handshake (U46): the deployed commit, the bundle's and serve.conf's hashes
  local commit=PLAN:DEPLOY.json-commit-on-the-box
  [ "${SERVE_PLAN:-0}" = 1 ] ||
    commit=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit"])' "$DEPLOY_DIR/DEPLOY.json")
  SERVE_ENV+=(TF_DS_DEPLOY="$commit" TF_DS_BUNDLE="$BUNDLE_SHA256" TF_DS_CONF="$(sha256sum "$SERVE_CONF" | cut -d' ' -f1)")
  # a follower's run dir (ranks 1-3), where first_light.sh down leaves its stop marker (exit 0 on a stop, 1 on a lost peer)
  [ "$RANK" != 0 ] && [ -n "${RUN:-}" ] && SERVE_ENV+=(TF_TFP_RUN="$RUN")
  if [ "$RANK" = 0 ]; then
    SERVE_ENV+=(TF_TP_WORLD="$WORLD" TF_TP_MASTER="$MASTER_IP" TF_TP_PORT="$TP_PORT" TF_DS_RANK_CACHE="$RANK_CACHE"
      TF_DS_KIT="$DS_KIT" TF_DS_ENGRAM="$ENGRAM_DIR" TF_DS_TOKEN_MAP="$TOKEN_MAP" TF_DS_STREAMS="$STREAMS"
      TF_DS_ROUND_ROWS="$ROUND_ROWS" TENSORFOLD_HIP_KERNELS="$KIT" TF_DS_HC_SIDE="$HC_SIDE" TF_DS_ENGRAM_AIO="$ENGRAM_AIO"
      TF_TOP_K_MAX="$TOP_K_MAX" TF_SEED_RANDOM="$SEED_RANDOM" TF_DS_L2_PREFETCH="$L2_PREFETCH" TF_DS_POOL="$POOL")
  fi
}

# serve_lanes_cmd: SERVE_CMD, tf-dsv41-lanes for this rank with the configuration's engine settings (rank 1's serving
# command; with rank 0 and --requests, the lane gate of the same configuration: qual/lane_gate.sh).
serve_lanes_cmd() {
  SERVE_CMD=("$BIN/tf-dsv41-lanes" "$MODEL_DIR" "$RANK_CACHE" "$RANK" "$WORLD" "$MASTER_IP" "$TP_PORT" "$DS_KIT"
    --pool "$POOL" --context "$CONTEXT" --streams "$STREAMS" --round-rows "$ROUND_ROWS" --drafts "$DRAFTS"
    --engram "$ENGRAM_DIR" --token-map "$TOKEN_MAP" --modalities "$MODALITIES" --hip-kit "$KIT"
    --arena-gib "$ARENA_GIB" --graphs "$GRAPHS" --side "$HC_SIDE" --aio "$ENGRAM_AIO" --prefetch "$L2_PREFETCH")
  [ -n "$ROUND_MS" ] && SERVE_CMD+=(--round-ms "$ROUND_MS")
  [ -n "$DRAFT_MS" ] && SERVE_CMD+=(--draft-ms "$DRAFT_MS")
  # (empty: the flag is left out, the engine's 4096: the argv of a build without it)
  [ "$MODALITIES" = text_image ] && [ -n "${IMAGE_TOKENS:-}" ] && SERVE_CMD+=(--image-tokens "$IMAGE_TOKENS")
  return 0
}

# serve_cmd: SERVE_CMD, the rank's argv. Rank 1's --pool, --context, --streams, --round-rows and --graphs are rank 0's
# TF_DS_POOL, --context, TF_DS_STREAMS, TF_DS_ROUND_ROWS and TF_DS_GRAPHS (upstream #25), from the same variables
# (tf-dsv41-lanes reads --graphs, --side, --aio and --prefetch, not TF_DS_GRAPHS, TF_DS_HC_SIDE, TF_DS_ENGRAM_AIO and
# TF_DS_L2_PREFETCH).
serve_cmd() {
  if [ "$RANK" = 0 ]; then
    SERVE_CMD=("$BIN/tensorfold-native" serve "$MODEL_DIR" --host "$SERVE_HOST" --port "$SERVE_PORT" --name "$SERVE_NAME"
      --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE"
      --prompt-cache-gib "$PROMPT_CACHE_GIB" --no-update-check)
    [ "$THINKING" = 1 ] && SERVE_CMD+=(--thinking) || SERVE_CMD+=(--no-thinking)
    # close goes as --loop-guard (the argv of a build without --loop-policy); stop and count as --loop-policy; off: neither
    case ${LOOP_POLICY:-off} in close) SERVE_CMD+=(--loop-guard);; stop|count) SERVE_CMD+=(--loop-policy "$LOOP_POLICY");; esac
    # (0: the flag is left out, the argv of a build without it)
    awk -v t="$DRAFT_TEMPERATURE" 'BEGIN { exit !(t > 0) }' && SERVE_CMD+=(--draft-temperature "$DRAFT_TEMPERATURE")
    [ "$DRAFTS" = 1 ] || SERVE_CMD+=(--no-drafts)
    # (empty: the flag is left out, the server's default)
    [ -z "$MAX_IMAGES" ] || SERVE_CMD+=(--vision-max-images "$MAX_IMAGES")
    # (empty or unset: the /cache routes are absent; the key file is read once, at the server's start)
    [ -z "${CACHE_TRANSFER_KEY:-}" ] || SERVE_CMD+=(--cache-transfer-key "$CACHE_TRANSFER_KEY")
  else
    serve_lanes_cmd
    [ -n "${LOGITS_SHA:-}" ] && SERVE_CMD+=(--logits-sha "$LOGITS_SHA")
  fi
  return 0
}

# serve_resolved FILE: the resolved configuration and this box's identities (what ran, from which bytes).
serve_resolved() {
  python3 - "$1" "$RANK" "$(hostname)" "$SERVE_CONF" "$DEPLOY_DIR" "$KIT" "$RANK_CACHE" "${SERVE_CMD[@]}" -- "${SERVE_ENV[@]}" <<'PY'
import hashlib, json, os, sys, time
out, rank, host, conf, deploy, kit, cache = sys.argv[1:8]
rest = sys.argv[8:]
cut = rest.index("--")
cmd, env = rest[:cut], rest[cut + 1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
dep = json.load(open(os.path.join(deploy, "DEPLOY.json")))
caches = sorted(n for n in os.listdir(cache) if n.startswith(f"rank{rank}of"))
json.dump({
    "schema": "tf-port-serve/1", "rank": int(rank), "host": host, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config": conf, "config_sha256": sha(conf), "deploy": {"dir": os.path.realpath(deploy), "commit": dep.get("commit"), "tree": dep.get("tree")},
    "binary_sha256": sha(cmd[0]), "kit_manifest_sha256": sha(os.path.join(kit, "manifest.json")), "rank_cache_files": caches,
    "argv": cmd, "env": dict(e.split("=", 1) for e in env),
}, open(out, "w"), indent=1)
PY
}

# serve_unit NAME DIR: the transient user unit's name for a run's process (SPAWN_UNIT=1): tfp-NAME-<run id>.
serve_unit() { echo "tfp-$1-$(basename "$2" | tr -dc '0-9A-Za-z' | tail -c 24)"; }

# serve_linger_check: a unit under the user manager only outlives the last login session when the user has Linger=yes;
# without it logind (KillUserProcesses=yes) stops user@UID.service 10 s after the last session ends and SIGKILLs every
# rank on the box (TP4-PEERDOWN ec91e930f8: a control-node reboot closed the last ssh session on D and killed rank 1).
# Refuse to start a unit then, and print the fix. The pod's workers get the linger marker at every boot from
# /etc/ltsp/ltsp.conf (POST_INIT_LINGER, card 7aa8487768); LINGER_USER names the user whose manager holds the units.
serve_linger_check() {
  local u=${LINGER_USER:-$(id -un)} l
  l=$(loginctl show-user "$u" -p Linger --value 2>/dev/null) || l=
  [ "$l" = yes ] && return 0
  die "SPAWN_UNIT=1 on $(hostname): user $u has Linger=${l:-unknown}, so its units (the ranks) are killed when its last session closes. Fix: run 'loginctl enable-linger $u' on this box (a user may set its own linger), check 'loginctl show-user $u -p Linger' prints Linger=yes, then start again"
}

# serve_unit_start UNIT LOG CMD...: CMD as the transient user unit UNIT (its output appended to LOG, cwd the log's
# directory), outside the caller's session: logind's KillUserProcesses=yes ends a session's processes with the session,
# a unit under the user manager stays. The unit's main process is CMD; nothing stops it but the caller (serve_stop,
# serve_lock_release), and a finished unit is collected.
serve_unit_start() {
  local unit=$1 log=$2; shift 2
  serve_linger_check
  systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
  systemd-run --user --quiet --collect --unit "$unit" -p WorkingDirectory="$(dirname "$log")" \
    -p StandardOutput=append:"$log" -p StandardError=append:"$log" -- "$@" ||
    die "systemd-run --user could not start unit $unit on $(hostname) (SPAWN_UNIT=1 needs a user manager)"
}

# serve_spawn PIDFILE LOG LOCKED: start SERVE_CMD with exactly SERVE_ENV in the background (on ENGINE_CPUS when set),
# under this box's GPU lock unless LOCKED is 1 (the caller holds it); PIDFILE gets the rank's own pid (not the lock
# wrapper's: a signal to it reaches the rank), written once the lock is held and the rank starts. SPAWN_UNIT=1: the
# rank runs as the transient user unit tfp-rank<R>-<run id> instead of a nohup'd child of this session.
serve_spawn() {
  local pidfile=$1 log=$2 locked=$3 lock=() pin=() wrap=() wv=WRAP$RANK
  [ "$locked" = 1 ] || lock=(flock -w "$LOCK_WAIT_S" /var/tmp/halo-gpu.lock)
  [ -n "${ENGINE_CPUS:-}" ] && pin=(taskset -c "$ENGINE_CPUS")
  [ -n "${!wv:-}" ] && read -ra wrap <<< "${!wv}"
  rm -f "$pidfile"
  local i
  if [ "$SPAWN_UNIT" = 1 ]; then
    local unit; unit=$(serve_unit "rank$RANK" "$(dirname "$pidfile")")
    : > "$log"
    serve_unit_start "$unit" "$log" "${lock[@]}" "$SERVE_TOOLS/pidexec.sh" "$pidfile" "${pin[@]}" env -i "${SERVE_ENV[@]}" "${wrap[@]}" "${SERVE_CMD[@]}"
    for i in $(seq 1 "$LOCK_WAIT_S"); do
      [ -s "$pidfile" ] && return 0
      systemctl --user is-active --quiet "$unit" || die "the rank did not start (unit $unit ended: lock wait or exec failed; $log)"
      sleep 1
    done
    systemctl --user stop "$unit" >/dev/null 2>&1 || true
    die "the GPU lock on $(hostname) was not free within ${LOCK_WAIT_S}s"
  fi
  # in the run's directory (the log's): the deployment is read-only, and a wrapper (rocprofv3) writes beside it
  (cd "$(dirname "$log")" && exec nohup "${lock[@]}" sh -c 'echo $$ > "$0"; exec "$@"' "$pidfile" "${pin[@]}" env -i "${SERVE_ENV[@]}" "${wrap[@]}" "${SERVE_CMD[@]}") </dev/null >"$log" 2>&1 &
  local wrapper=$!
  for i in $(seq 1 "$LOCK_WAIT_S"); do
    [ -s "$pidfile" ] && return 0
    kill -0 "$wrapper" 2>/dev/null || die "the rank did not start (lock wait or exec failed; $log)"
    sleep 1
  done
  die "the GPU lock on $(hostname) was not free within ${LOCK_WAIT_S}s"
}

# serve_latest: the latest run's id on this box (RUN_ROOT/latest points at it), or nothing.
serve_latest() { basename "$(readlink "$RUN_ROOT/latest" 2>/dev/null)" 2>/dev/null || true; }

# serve_pid FILE: the live pid in FILE, or nothing.
serve_pid() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null && cat "$1"; return 0; }

# serve_stop PIDFILE NAME: SIGTERM, then SIGKILL after STOP_TIMEOUT_S.
serve_stop() {
  local pid; pid=$(serve_pid "$1")
  [ -n "$pid" ] || { echo "$2: not running"; return 0; }
  kill -TERM "$pid"
  local i
  for i in $(seq 1 "$STOP_TIMEOUT_S"); do kill -0 "$pid" 2>/dev/null || { echo "$2: stopped ($pid)"; return 0; }; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
  echo "$2: killed after ${STOP_TIMEOUT_S}s ($pid)"
}

# serve_lock_hold DIR [WAIT_S]: hold this box's GPU lock for the run (a holder process; its pid in DIR/lock.pid once
# held), waiting at most WAIT_S (default LOCK_WAIT_S); a holder that did not get it in time is taken down, never left.
# The holder (lock_watch.sh) lets the lock go when this box's rank (DIR/rank<R>.pid) ends, also when nothing stopped it.
serve_lock_hold() {
  local dir=$1 wait=${2:-$LOCK_WAIT_S} i
  rm -f "$dir/lock.pid"
  [ -n "$(serve_pid "$dir/rank$RANK.pid")" ] || rm -f "$dir/rank$RANK.pid"  # an earlier rank's, in a reused run dir
  if [ "$SPAWN_UNIT" = 1 ]; then  # the holder as a unit (tfp-lock<R>-<run id>): it must outlive this session too
    local unit; unit=$(serve_unit "lock$RANK" "$dir")
    serve_unit_start "$unit" "$dir/lock.log" flock -w "$wait" /var/tmp/halo-gpu.lock "$SERVE_TOOLS/pidexec.sh" "$dir/lock.pid" \
      "$SERVE_TOOLS/lock_watch.sh" "$dir/rank$RANK.pid"
    for i in $(seq 1 "$wait"); do
      [ -s "$dir/lock.pid" ] && return 0
      systemctl --user is-active --quiet "$unit" || break
      sleep 1
    done
    [ -s "$dir/lock.pid" ] && return 0
    systemctl --user stop "$unit" >/dev/null 2>&1 || true; rm -f "$dir/lock.pid"
    die "the GPU lock on $(hostname) was not free within ${wait}s"
  fi
  nohup flock -w "$wait" /var/tmp/halo-gpu.lock sh -c "echo \$\$ > '$dir/lock.pid'; exec '$SERVE_TOOLS/lock_watch.sh' '$dir/rank$RANK.pid'" \
    </dev/null >/dev/null 2>&1 &
  local holder=$!
  for i in $(seq 1 "$wait"); do
    [ -s "$dir/lock.pid" ] && return 0
    kill -0 "$holder" 2>/dev/null || break
    sleep 1
  done
  [ -s "$dir/lock.pid" ] && return 0
  # flock gave up, or got the lock just now: its child (the sleeping holder) and flock itself go
  pkill -P "$holder" 2>/dev/null; kill "$holder" 2>/dev/null; rm -f "$dir/lock.pid"
  die "the GPU lock on $(hostname) was not free within ${wait}s"
}

serve_lock_release() { local pid; pid=$(serve_pid "$1/lock.pid"); [ -n "$pid" ] && kill "$pid"; rm -f "$1/lock.pid"; return 0; }
