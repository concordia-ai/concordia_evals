from pathlib import Path

from platformdirs import user_cache_dir

CONCORDIA_EVALS_CACHE_PATH = Path(user_cache_dir("concordia-evals"))
