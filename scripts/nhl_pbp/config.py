from __future__ import annotations
import os
from pathlib import Path

# --- Load .env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()  # search upward from CWD
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        f = p / ".env"
        if f.exists():
            load_dotenv(f, override=True)
            break
except Exception:
    pass

API_BASE: str = os.getenv("NHL_API_BASE", "https://api-web.nhle.com/v1")

def _find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.py").exists():
            return p
    # fallback: assume .../scripts/<this file>
    return start.parents[1]

PROJECT_ROOT = _find_repo_root(Path(__file__).resolve())

# Default target: <repo>/ift6758/data/nhl
DEFAULT_CACHE_DIR = PROJECT_ROOT / "ift6758" / "data" / "nhl"

def _resolve_cache_dir(env_val: str | None) -> Path:
    if not env_val or not env_val.strip():
        return DEFAULT_CACHE_DIR
    raw = env_val.strip().strip('"').strip("'")
    p = Path(os.path.expanduser(raw))

    # If the value is absolute with a drive (Windows) or POSIX absolute, keep as-is.
    if p.is_absolute() and (p.drive or os.name != "nt"):
        return p

    # Treat everything else (including Git-Bash-style "/something") as REPO-RELATIVE.
    rel = raw.lstrip("/\\")
    return (PROJECT_ROOT / rel).resolve()

RAW_DIR_PATH: Path = _resolve_cache_dir(os.getenv("NHL_CACHE_DIR"))
RAW_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Export absolute for any code that reads env directly later
os.environ["NHL_CACHE_DIR"] = str(RAW_DIR_PATH)

# Back-compat names
RAW_DIR: str = str(RAW_DIR_PATH)
CACHE_DIR: str = RAW_DIR

REQUEST_PAUSE_SEC: float = float(os.getenv("NHL_REQUEST_PAUSE", "0.25"))
TIMEOUT_SEC: int = int(os.getenv("NHL_TIMEOUT_SEC", "20"))
MAX_RETRIES: int = int(os.getenv("NHL_MAX_RETRIES", "5"))
SHOW_PROGRESS: bool = os.getenv("NHL_PROGRESS", "1") not in {"0","false","False","no","No"}
REQUIRED_SEASONS = tuple(range(2016, 2024))