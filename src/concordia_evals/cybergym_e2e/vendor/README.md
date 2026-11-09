# Vendored upstream files

`validate.py` is a byte-identical copy of `scripts/validate.py` from
https://github.com/sunblaze-ucb/cybergym-e2e at pinned commit
`d0ae0f02e76ebc084389bf21e755f4b41ac9765a` (Apache License 2.0, copyright the
CyberGym-E2E authors). It is deployed into task containers and executed there;
keeping it byte-identical preserves the benchmark's scoring semantics.
Verify with:

```bash
curl -sL "https://raw.githubusercontent.com/sunblaze-ucb/cybergym-e2e/d0ae0f02e76ebc084389bf21e755f4b41ac9765a/scripts/validate.py" | diff - validate.py
```
