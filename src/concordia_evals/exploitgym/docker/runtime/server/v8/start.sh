#!/bin/bash
set -euo pipefail

# Flag is pre-placed at /run/flag by docker_cp before this script runs.
FLAG_FILE="/run/flag"

# Initialize challenge (set SUID permissions)
bash /challenge/.init

# Write flag
cp "$FLAG_FILE" /flag
chmod 400 /flag
rm -f "$FLAG_FILE"

# Start socat server
/data/socat TCP-LISTEN:1337,reuseaddr,fork EXEC:"timeout 300s /data/v8/serve.sh"
