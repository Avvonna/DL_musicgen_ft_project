import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dir_utils import META_DIR
from utils.requests_utils import retry_call

load_dotenv()

# Я хотел попробовать локально поднять модельку, но она отрабатывает медленнее чем OR API
# Поэтому оставил как было в итоге, поменяв модель на "qwen/qwen3.5-9b"
USE_LOCAL = False  # True = ollama, False = OpenRouter


if USE_LOCAL:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    MODEL = "qwen3:8b"
    EXTRA_BODY = {
        "keep_alive": -1,
        "think": False,
    }
else:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    MODEL = "qwen/qwen3.5-9b"
    EXTRA_BODY = {"reasoning": {"enabled": False}}


TEMPERATURE = 0.3
PROCESSED_DIR = f"dataset/{MODEL.split('/')[-1]}"
os.makedirs(PROCESSED_DIR, exist_ok=True)


if USE_LOCAL:
    SYSTEM_PROMPT = """
    You extract music metadata from the provided caption and aspects.

    Rules:
    - Use only information explicitly supported by the caption or aspects.
    - Do not guess or add typical genre/instrument details.
    - Fill every field.
    - Use "None" if a value is not supported.
    - Keep outputs short and specific.

    Field rules:
    - description: 1-2 English sentences summarizing only stated musical details
    - general_mood: 1-4 comma-separated mood adjectives
    - genre_tags: 2-5 lowercase genre/style tags, or empty list if unsupported
    - lead_instrument: most prominent instrument/sound source, or "None"
    - accompaniment: 1 short comma-separated phrase of backing elements, or "None"
    - tempo_and_rhythm: short phrase, e.g. "slow, laid-back groove"
    - vocal_presence: short phrase, e.g. "no vocals", "female vocal", "spoken word"
    - production_quality: short phrase, e.g. "lo-fi", "polished studio", "raw live"

    Return valid JSON only.
    """.strip()
else:
    SYSTEM_PROMPT = """
    You extract structured music metadata from a caption and aspect list.

    Use only information explicitly supported by the input.
    Do not guess missing details.
    Fill every field and use "None" when unsupported.
    Keep values short, specific, and schema-compliant.
    """.strip()

class Response(BaseModel):
    """Требуемый формат ответа модели"""
    description: str = Field(
        default="",
        description="Concise natural-language summary of the track (1-3 sentences)."
    )
    general_mood: str = Field(
        default="None",
        description="Overall emotional tone, e.g. uplifting, melancholic, tense, calm."
    )
    genre_tags: list[str] = Field(
        default_factory=list,
        description="2-5 genre/style tags in lowercase."
    )
    lead_instrument: str = Field(
        default="None",
        description="Most prominent instrument or sound source."
    )
    accompaniment: str = Field(
        default="None",
        description="Main backing instruments or textures."
    )
    tempo_and_rhythm: str = Field(
        default="None",
        description="Tempo and rhythmic feel description."
    )
    vocal_presence: str = Field(
        default="None",
        description='Description of vocal presence, e.g. "male vocalist singing melodically", "female choir", "no vocals".'
    )
    production_quality: str = Field(
        default="None",
        description="Recording and production character description."
    )


def clean_response(raw: str) -> str:
    """Удаляет markdown-обёртки вокруг JSON."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def call_model(prompt: str) -> Response:
    if USE_LOCAL:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            extra_body=EXTRA_BODY,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Response",
                    "schema": Response.model_json_schema(),
                    "strict": True,
                },
            },
        )
        raw = response.choices[0].message.content or ""
        raw = clean_response(raw)
        return Response.model_validate_json(raw)
    else:
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            extra_body=EXTRA_BODY,
            response_format=Response,
        )
        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError("Empty response from model")
        return result


meta_files = sorted(glob.glob(os.path.join(META_DIR, "*.json")))

for meta_file in tqdm(meta_files, desc="Enrichment process"):
    ytid = os.path.basename(meta_file)
    enriched_path = os.path.join(PROCESSED_DIR, ytid)

    if os.path.exists(enriched_path):
        continue

    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aspects = data.get('aspect_list') or []
    if isinstance(aspects, str):
        try:
            aspects = ast.literal_eval(aspects)
        except Exception:
            aspects = [aspects]
    caption = data.get('caption') or ""

    user_prompt = (
        "Extract music metadata from the following input.\n"
        "Use only the provided caption and aspects.\n"
        "If a field is unsupported, return 'None'.\n\n"
        f"Caption: {caption}\n"
        f"Aspects: {', '.join(map(str, aspects))}"
    )

    def _call(prompt=user_prompt):
        return call_model(prompt)

    try:
        res = retry_call(_call)
        if res is None:
            raise ValueError("Model returned no parsed object")
        if not res.description:
            raise ValueError("No description provided")
    except Exception as e:
        print(f"Error enriching {ytid}: {e}")
        continue

    # Сохраняем
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(res.model_dump(), f, ensure_ascii=False, indent=2)
