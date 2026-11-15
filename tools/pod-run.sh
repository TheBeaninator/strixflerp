#!/usr/bin/env bash
# Run a command on control-node inside /data/projects/strixflerp. Usage: tools/pod-run.sh 'cmd ...'
# Long jobs: wrap in `timeout` yourself; output streams back.
set -euo pipefail
printf 'cd /data/projects/strixflerp && %s\n' "$*" | ssh -A relay-host ssh pod@control-node bash -s
