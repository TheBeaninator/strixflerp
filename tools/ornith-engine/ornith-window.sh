#!/usr/bin/env bash
# The window engine's UP/DOWN lines in engines.conf (pod-engine serving ornith397_flerp). Runs on control-node.
#   ornith-window.sh up [--minutes N] [--set K=V ...]   queue ONE win_run job (F D A C); blocks until it ends. Refuses (exit 3,
#                                                       "window pending") while another job of this engine is queued or open.
#   ornith-window.sh url        open window: prints the url, exit 0. Queued or loading: "window pending ..." on stderr, exit 2.
#                               None: exit 1. Callers cap what they queue on exit 2 and answer at once instead of holding a slot.
#   ornith-window.sh down       operator stop (stop file); win_run then closes the window. Not a caller's idle timer.
# The window's end is the job's own win_run time limit (`for` = minutes + 25) plus engine.py --max-min, so the keeper always
# restores the nodes. The first real `up` needs owner's go (card 83a59b0e07).
set -eu
S=/data/pod/winjob/ornith-engine; mkdir -p $S
case "${1:-}" in
  up) shift
      if [ -s $S/pending.json ] || [ -s $S/endpoint.json ]; then echo "window pending: $(cat $S/pending.json 2>/dev/null)" >&2; exit 3; fi
      f=$(mktemp /tmp/ornith-win-XXXXXX.json)
      python3 /data/projects/strixflerp-cfg/tools/ornith-engine/win_args.py run "$@" > "$f"
      printf '{"queued": "%s", "args": "%s"}\n' "$(date -u +%FT%TZ)" "$*" > $S/pending.json
      trap 'rm -f $S/pending.json $S/endpoint.json "$f"' EXIT
      pod-winjob win_run --args "$f" ;;
  down) : > $S/stop ;;
  url) if [ -s $S/endpoint.json ]; then python3 -c "import json;print(json.load(open('$S/endpoint.json'))['url'])"
       elif [ -s $S/pending.json ]; then echo "window pending: $(cat $S/pending.json)" >&2; exit 2
       else exit 1; fi ;;
  *) echo "usage: ornith-window.sh up|down|url" >&2; exit 2 ;;
esac
