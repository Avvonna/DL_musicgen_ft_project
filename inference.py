import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "audiocraft"))
from audiocraft.models import MusicGen  # type: ignore  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--trained_duration", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()

def json_to_prompt(data: dict) -> str:
    description = data.get("description", "").strip()

    extra_fields = [
        "general_mood",
        "genre_tags",
        "lead_instrument",
        "accompaniment",
        "tempo_and_rhythm",
        "vocal_presence",
        "production_quality",
    ]

    parts = []
    if description:
        parts.append(description.rstrip("."))

    for key in extra_fields:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(x) for x in value if str(x).strip())
        else:
            value = str(value).strip()
        if value:
            parts.append(f"{key}: {value}")

    return ". ".join(parts)

def load_prompts(json_dir: Path):
    items = []
    for path in sorted(json_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompt = json_to_prompt(data)
        items.append((path.stem, prompt))
    return items

def main():
    args = parse_args()
    prompts = load_prompts(args.json_dir)

    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.max_duration = args.trained_duration
    model.set_generation_params(
        duration=args.duration,
        use_sampling=True,
        top_k=250,
        top_p=0.0,
        temperature=1.05,
        cfg_coef=3.0,
        extend_stride=max(1, args.trained_duration // 2),
    )

    if args.finetuned:
        if args.checkpoint is None:
            raise ValueError("Нужен --checkpoint при --finetuned")
        state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        model.lm.load_state_dict(state["best_state"]["model"], strict=False)

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        stems = [x[0] for x in batch]
        descriptions = [x[1] for x in batch]

        with torch.no_grad():
            wavs = model.generate(descriptions, progress=True)

        for wav, stem in zip(wavs, stems):
            sf.write(
                str(args.json_dir / f"{stem}.wav"),
                wav[0].cpu().numpy(),
                32000,
            )


if __name__ == "__main__":
    main()
