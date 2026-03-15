from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent

PROJECT_DIR = _UTILS_DIR.parent

AUDIOCRAFT_DIR = PROJECT_DIR / "audiocraft"

META_DIR = PROJECT_DIR / "dataset" / "meta"
RAW_DATA_DIR = PROJECT_DIR / "dataset" / "raw"

MANIFESTS_DIR = PROJECT_DIR / "manifests"
RES_DIR = PROJECT_DIR / "RES"

def ensure_directories_exist():
    META_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

ensure_directories_exist()
