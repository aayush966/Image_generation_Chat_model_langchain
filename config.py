"""Central configuration, loaded from the environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or ""
).strip()

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()
# Public Spaces that serve the same open-source models without a token.
# Each entry is tried in order; the first that answers wins.
SPACE_MAP = {
    "black-forest-labs/FLUX.1-schnell": [
        "black-forest-labs/FLUX.1-schnell",
        "multimodalart/FLUX.1-schnell",
    ],
}

_BACKEND = os.getenv("BACKEND", "auto").strip().lower()
# "auto": use the Inference API when a token exists, otherwise fall back to the
# public Space so the app still produces real images out of the box.
BACKEND = _BACKEND if _BACKEND != "auto" else ("api" if HF_TOKEN else "space")

OUTPUT_DIR = ROOT / os.getenv("OUTPUT_DIR", "generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models that only need a handful of steps (turbo / schnell distilled variants).
FAST_MODELS = ("schnell", "turbo", "lightning", "lcm")


def default_steps(model_id: str = IMAGE_MODEL) -> int:
    return 4 if any(tag in model_id.lower() for tag in FAST_MODELS) else 28


def default_guidance(model_id: str = IMAGE_MODEL) -> float:
    return 0.0 if any(tag in model_id.lower() for tag in FAST_MODELS) else 7.0
