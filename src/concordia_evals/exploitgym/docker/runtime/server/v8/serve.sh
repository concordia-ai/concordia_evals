#!/bin/bash
# socat handler: reads JS from stdin, runs via /challenge/run
MAX_SIZE=10485760
MAX_SIZE_1=$((MAX_SIZE + 1))

tmpfile=$(mktemp /tmp/exploit-XXXXXX.js)
head -c "$MAX_SIZE_1" > "$tmpfile"
chmod 644 "$tmpfile"

actual=$(stat -c%s "$tmpfile")
if [ "$actual" -ge "$MAX_SIZE_1" ]; then
    echo "Error: file too large (max ${MAX_SIZE} bytes)"
    rm -f "$tmpfile"
    exit 1
fi

/challenge/run "$tmpfile"
rm -f "$tmpfile"
