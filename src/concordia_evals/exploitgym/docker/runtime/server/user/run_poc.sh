#!/bin/bash

ulimit -c 0

export ASAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_abort=0:disable_coredump=0:abort_on_error=1
export UBSAN_OPTIONS=handle_segv=0:halt_on_error=1:abort_on_error=1

cd /var/empty/nobody

POC="$1"

if nm ${BINARY_PATH} | grep -q __afl_area_ptr ; then
    exec ${BINARY_PATH} "$POC"
else
    exec ${BINARY_PATH} -handle_segv=0 -handle_abrt=0 -verbosity=0 "$POC"
fi
