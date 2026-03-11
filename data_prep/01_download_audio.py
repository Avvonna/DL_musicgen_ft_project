import json
import os
import subprocess
from typing import Any, cast

import yt_dlp
from datasets import load_dataset
from tqdm import tqdm
from yt_dlp.utils import DownloadError

# Настройки
DATA_DIR = "dataset"
MAX_SAMPLES = 1_000
os.makedirs(DATA_DIR, exist_ok=True)

# Загрузка датасета
ds = load_dataset("google/MusicCaps", split="train")

# Настройки для yt-dlp
ydl_opts = {
    'format': 'bestaudio/best',
    'quiet': True,
    'no_warnings': True,
    'extract_flat': False,
    'sleep_interval': 5,
    'max_sleep_interval': 15,
    'sleep_interval_requests': 1,
}

success_count = 0

with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
    pbar = tqdm(total=MAX_SAMPLES, desc="Скачивание аудио")

    for item in ds:
        if success_count >= MAX_SAMPLES:
            break

        row = cast(dict[str, Any], item)

        ytid = row['ytid']
        start_s = row['start_s']
        end_s = row['end_s']
        duration = end_s - start_s

        wav_path = os.path.join(DATA_DIR, f"{ytid}.wav")
        json_path = os.path.join(DATA_DIR, f"{ytid}_meta.json")

        # Пропускаем, если уже скачали
        if os.path.exists(wav_path) and os.path.exists(json_path):
            success_count += 1
            pbar.update(1)
            continue

        youtube_url = f"https://www.youtube.com/watch?v={ytid}"

        try:
            # Получаем прямую ссылку на аудиопоток
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info.get("url")

            # Скачиваем ровно нужный фрагмент
            # -ss: начало, -i: входной поток, -t: длительность, -ar 32000 -ac 1
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5',
                '-ss', str(start_s),
                '-i', stream_url,
                '-t', str(duration),
                '-ar', '32000', '-ac', '1', '-c:a', 'pcm_s16le',
                wav_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            # Сохраняем оригинальные метаданные
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "ytid": ytid,
                    "caption": row['caption'],
                    "aspect_list": row['aspect_list']
                }, f, ensure_ascii=False, indent=2)

            success_count += 1
            pbar.update(1)

        except DownloadError as _:
            pass
        except Exception as e:
            print(f"\nОшибка {ytid}: {e}")
            continue

print(f"Успешно скачано {success_count} файлов.")
