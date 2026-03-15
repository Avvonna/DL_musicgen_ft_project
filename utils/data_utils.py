import gzip
import json
import shutil
from pathlib import Path

from utils.dir_utils import PROJECT_DIR


def write_jsonl_gz(entries, filepath: Path):
    """Создание файла манифеста"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def copy_configs(paths_map: list[tuple[str, str]]):
    """Копирует файлы по относительным путям проекта."""
    for src, dst in paths_map:
        src_path = PROJECT_DIR / src
        dst_path = PROJECT_DIR / dst
        shutil.copy2(src_path, dst_path)
