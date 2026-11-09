#!/bin/bash
set -euo pipefail
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MAX_SIZE=10485760  # 10MB
BINARY_PATH="$1"

UPLOAD_PATH=$(mktemp -p /tmp "upload_${TIMESTAMP}_XXXXXX") || {
    echo "ERROR: failed to create temp file"
    exit 1
}

echo "------------------------------------------------------------------"
echo "Input format: '<eight_char_ascii_hex_size><file bytes>'"
echo "Max allowed size: $MAX_SIZE bytes"
echo "After sending the input, you can interact with the target binary over the same connection."
echo "------------------------------------------------------------------"

# Protocol: client sends "<hex size><file bytes>"
len_be_hex=$(dd bs=1 count=8 2>/dev/null) || true  # Don't fail on short read; we'll check length later

if [ "${#len_be_hex}" -ne 8 ]; then
  echo "ERROR: short header"
  exit 1
fi

# Check if it's valid hex
if ! [[ "$len_be_hex" =~ ^[0-9a-fA-F]{8}$ ]]; then
    echo "ERROR: invalid hex header"
    exit 1
fi

echo "[*] Received length (hex, big-endian): $len_be_hex"

filesize=$(( 16#${len_be_hex:0:2}${len_be_hex:2:2}${len_be_hex:4:2}${len_be_hex:6:2} ))
echo "[*] Received file size: $filesize bytes"

# Validate: format and reasonable size (e.g., < 10MB)
if ! [[ "$filesize" =~ ^[0-9]+$ ]] || [ "$filesize" -le 0 ] || [ "$filesize" -gt $MAX_SIZE ]; then
    echo "ERROR: invalid size"
    exit 1
fi

# Read exactly $filesize bytes into file
dd bs=1 count="$filesize" of="$UPLOAD_PATH" 2>/dev/null

# Now exec the fuzzer: its stdin/stdout become the socket
export BINARY_PATH
exec /data/user/run_poc.sh "$UPLOAD_PATH"
