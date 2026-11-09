#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <binary_path> [target]"
    exit 1
fi

BINARY_PATH="$1"
TARGET="${2:-EXEC}"

source /data/user/init.sh ${TARGET}


/data/socat TCP-LISTEN:8000,reuseaddr,fork \
    EXEC:"env HOME=/var/empty/nobody timeout 300s /data/user/handler.sh $BINARY_PATH",su=nobody
