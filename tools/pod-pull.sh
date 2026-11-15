#!/usr/bin/env bash
# Pull gate results written on control-node back into this tree. Usage: tools/pod-pull.sh [subdir, default gate/results]
set -euo pipefail
cd "$(dirname "$0")/.."
d="${1:-gate/results}"; mkdir -p "$d"
rsync -a -e "ssh -A relay-host ssh" "pod@control-node:/data/projects/strixflerp/$d/" "$d/"
