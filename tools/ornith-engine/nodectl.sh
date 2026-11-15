#!/usr/bin/env bash
# Node side of the Ornith 397B window engine (rides in the WINJOB kit, runs from $KIT_DIR on a worker).
#   nodectl.sh up UNIT [ENV=VAL ...] -- BIN ARGS...   start BIN (relative to this kit) as the transient user unit UNIT
#   nodectl.sh down UNIT                              stop it (SIGTERM; flerp ranks close cleanly on SIGTERM)
#   nodectl.sh status UNIT                            active state + the last JSON events
#   nodectl.sh serving                                print serving-looking processes (the engine refuses a node that serves)
# Everything runs as a user unit named flerp-ornith-*, so win_run's cleanup and `down` find it. Weights are not in
# the kit: ranks read /cache/flerp-p5/*.staged (PLACE) or stream from a rank origin over the rail.
set -u
KIT=$(cd "$(dirname "$0")/.." && pwd)
export XDG_RUNTIME_DIR=/run/user/$(id -u)
LOGD=$HOME/flerp-ornith-run; mkdir -p "$LOGD"
case "${1:-}" in
  up)
    unit=flerp-ornith-$2; shift 2
    envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("--setenv=$1"); shift; done
    shift; bin=$1; shift
    systemctl --user stop "$unit" 2>/dev/null; systemctl --user reset-failed "$unit" 2>/dev/null
    exec systemd-run --user --unit "$unit" --collect --setenv=LD_LIBRARY_PATH="$KIT/lib" "${envs[@]}" \
      -p StandardOutput=file:"$LOGD/$unit.jsonl" -p StandardError=file:"$LOGD/$unit.err" -- "$KIT/$bin" "$@" ;;
  down)
    unit=flerp-ornith-$2; systemctl --user stop "$unit"; systemctl --user is-active "$unit" >/dev/null 2>&1 && echo "still active" && exit 1; exit 0 ;;
  status)
    unit=flerp-ornith-$2; echo "state $(systemctl --user is-active "$unit" 2>&1)"; tail -n 3 "$LOGD/$unit.jsonl" 2>/dev/null | cut -c1-300 ;;
  serving)
    pgrep -af "tensorfold-native|tf-dsv41-|strix_|strixite|gufo|sglang|llama-server|ds4|vllm" | grep -v -e pgrep -e nodectl | cut -c1-120 ;;
  *) echo "usage: nodectl.sh up|down|status|serving ..." >&2; exit 2 ;;
esac
