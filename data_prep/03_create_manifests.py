import random
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from utils.data_utils import write_jsonl_gz
from utils.dir_utils import MANIFESTS_DIR, PROJECT_DIR, RAW_DATA_DIR

JSON_DIR = PROJECT_DIR / "dataset" / "qwen3.5-9b"

SAMPLE_RATE = 32000
TRAIN_RATIO = 0.9
MIN_DURATION = 8.0
SEED = 42

def prepare_train_val_data():
    json_files = sorted(JSON_DIR.glob("*.json"))
    all_entries = []
    skipped = 0

    if not json_files:
        raise FileNotFoundError(f"Не найдено json-файлов в {JSON_DIR}")

    random.seed(SEED)

    for json_file in tqdm(json_files, desc="Сборка train/valid/evaluate"):
        ytid = json_file.stem
        wav_path = RAW_DATA_DIR / f"{ytid}.wav"

        if not wav_path.exists():
            continue

        try:
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate

            if duration < MIN_DURATION:
                skipped += 1
                continue

            entry = {
                "path": wav_path.as_posix(),
                "duration": float(duration),
                "sample_rate": int(info.samplerate),
                "amplitude": None,
                "weight": None,
                "info_path": str(json_file).replace("\\", "/")
            }
            all_entries.append(entry)

        except Exception as e:
            print(f"\nОшибка чтения аудио {wav_path}: {e}")

    if not all_entries:
        raise RuntimeError("Не удалось собрать ни одной записи для train/valid")

    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * TRAIN_RATIO)

    train_entries = all_entries[:split_idx]
    valid_entries = all_entries[split_idx:]

    manifests_dir = Path(MANIFESTS_DIR)
    train_path = manifests_dir / "train.jsonl.gz"
    valid_path = manifests_dir / "valid.jsonl.gz"

    write_jsonl_gz(train_entries, train_path)
    write_jsonl_gz(valid_entries, valid_path)

    return train_entries, valid_entries, all_entries, skipped

def main():
    train_entries, valid_entries, all_entries, skipped = prepare_train_val_data()

    print(
        f"Выборка разбита: "
        f"train - {len(train_entries)}; "
        f"valid - {len(valid_entries)}; "
        f"total - {len(all_entries)} (of {len(all_entries) + skipped})"
    )


if __name__ == "__main__":
    main()
