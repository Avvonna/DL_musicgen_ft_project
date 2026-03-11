import glob
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.requests_utils import retry_call

load_dotenv()

# Настройки OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "arcee-ai/trinity-large-preview:free"
TEMPERATURE = 0.3

DATA_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/enriched"
os.makedirs(PROCESSED_DIR, exist_ok=True)


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


SYSTEM_PROMPT = """
You are an expert musicologist and metadata tagger.
I will give you a raw text description and a list of aspects of a music track.
Return data strictly in the requested schema.
"""


class Response(BaseModel):
    """Требуемый формат ответа модели"""
    description: str = Field(
        default="",
        description="Concise natural-language summary of the track (1–3 sentences)."
    )

    general_mood: str | None = Field(
        default=None,
        description="Overall emotional tone of the music, e.g. uplifting, melancholic, tense, calm. Use null if unclear."
    )

    genre_tags: list[str] = Field(
        default_factory=list,
        description="List of genre or style tags such as rock, jazz, ambient, electronic. Return an empty list if unclear."
    )

    lead_instrument: str | None = Field(
        default=None,
        description="Most prominent lead instrument or sound source. Use null if unclear."
    )

    accompaniment: str | None = Field(
        default=None,
        description="Main backing instruments or textures supporting the lead instrument. Use null if unclear."
    )

    tempo_and_rhythm: str | None = Field(
        default=None,
        description="Description of tempo and rhythmic feel, e.g. slow waltz, fast syncopated beat, steady mid-tempo groove. Use null if unclear."
    )

    vocal_presence: Literal[
        "instrumental",
        "male vocals",
        "female vocals",
        "mixed vocals",
        "choir",
        "spoken word",
        "unclear"
    ] = Field(
        default="unclear",
        description="Type of vocal presence. Choose exactly one label."
    )

    production_quality: str | None = Field(
        default=None,
        description="Description of recording and production character, e.g. lo-fi, polished studio, live recording, heavily processed. Use null if unclear."
    )


meta_files = glob.glob(os.path.join(DATA_DIR, "*_meta.json"))

for meta_file in tqdm(meta_files, desc="Enrichment process"):
    ytid = os.path.basename(meta_file).replace("_meta.json", "")
    enriched_path = os.path.join(PROCESSED_DIR, f"{ytid}.json")

    # Пропускаем, если уже обработали
    if os.path.exists(enriched_path):
        continue

    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_prompt = (
        f"Caption: {data['caption']}\n"
        f"Aspects: {', '.join(data['aspect_list'])}"
    )

    def _call():
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            response_format=Response
        )
        return response.choices[0].message.parsed

    try:
        res = retry_call(_call)
        if res is None:
            raise ValueError("Model returned no parsed object")
    except Exception as e:
        print(f"Error enriching {ytid}: {e}")
        continue

    # Сохраняем
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(res.model_dump(), f, ensure_ascii=False, indent=2)
