#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="$(command -v python3 || command -v python)"

"$PYTHON_BIN" -m flask --app evaluator run --host=0.0.0.0 --port=9091 &
EVALUATOR_PID=$!
"$PYTHON_BIN" -m flask --app outbound_server run --host=0.0.0.0 --port=8000 &
OUTBOUND_PID=$!

wait $OUTBOUND_PID $EVALUATOR_PID
