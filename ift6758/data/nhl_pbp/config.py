from __future__ import annotations
import os
from pathlib import Path

# --- Load .env robustly (works from any working dir and nesting depth) ---
try:
    from dotenv import load_dotenv

    # 1) Load from current working directory (repo root in most cases)
    load_dotenv()  # looks for .env in CWD and up the tree

    # 2) Also try walking up from THIS file to find a .env at repo root
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        cand = p / ".env"
        if cand.exists():
            load_dotenv(cand, override=True)
            break
except Exception:
    # dotenv is optional; if missing, we just skip
    pass

# --- Config values (read after .env is loaded) ---
API_BASE: str = os.getenv("NHL_API_BASE", "https://api-web.nhle.com/v1")
RAW_DIR: str = os.getenv("NHL_CACHE_DIR", "nhl/pub/raw")
CACHE_DIR = os.getcwd() + RAW_DIR
print(CACHE_DIR)
REQUEST_PAUSE_SEC: float = float(os.getenv("NHL_REQUEST_PAUSE", "0.25"))
TIMEOUT_SEC: int = int(os.getenv("NHL_TIMEOUT_SEC", "20"))
MAX_RETRIES: int = int(os.getenv("NHL_MAX_RETRIES", "5"))
SHOW_PROGRESS: bool = os.getenv("NHL_PROGRESS", "1") not in {"0","false","False","no","No"}
# Seasons required by the assignment (inclusive)
REQUIRED_SEASONS = tuple(range(2016, 2024))  # 2016-17 ... 2023-24