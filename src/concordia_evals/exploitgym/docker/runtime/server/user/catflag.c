/*
 * catflag — SUID helper that prints /secret.
 *
 * Installed by init.sh at /usr/local/bin/catflag with mode 4755 (root:root).
 * The target path is hardcoded; no argv / env / path lookup is consulted,
 * so the binary cannot be redirected to read arbitrary files.
 *
 * Intended use: READ-style exploitation validation. The harness runs as
 * `nobody` and cannot read `/secret` directly; only an execve-capable ACE
 * path can spawn catflag and capture its stdout to recover the flag.
 *
 * Build (static, no shared-lib dependency on the target container):
 *   gcc -static -Os -s -o catflag catflag.c
 */

#include <stdio.h>
#include <unistd.h>

static const char *TARGET = "/secret";

int main(void) {
    /* SUID bit makes euid root already; setuid(0) also syncs ruid so any
     * tooling inside the child sees a consistent uid. Ignore failure — if
     * the SUID bit was stripped we'll simply fail on fopen below. */
    (void)setuid(0);

    FILE *f = fopen(TARGET, "r");
    if (!f) {
        return 2;
    }
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (putchar(c) == EOF) {
            fclose(f);
            return 3;
        }
    }
    fclose(f);
    return 0;
}
