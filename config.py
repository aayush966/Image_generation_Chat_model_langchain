"""Central configuration, loaded from the environment / .env file."""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or ""
).strip()

# --------------------------------------------------------------------------- #
# Google Vertex AI (paid): Nano Banana images and Veo video
# --------------------------------------------------------------------------- #
# A service-account JSON key; relative paths are resolved against the project.
_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "key.json").strip()
GOOGLE_CREDENTIALS = Path(_creds) if Path(_creds).is_absolute() else ROOT / _creds
if GOOGLE_CREDENTIALS.is_file():
    # The Google client libraries read the key from this variable.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GOOGLE_CREDENTIALS)


def _key_project() -> str:
    try:
        return json.loads(GOOGLE_CREDENTIALS.read_text(encoding="utf-8")).get("project_id", "")
    except (OSError, ValueError):
        return ""


GCP_PROJECT = (os.getenv("GCP_PROJECT") or _key_project()).strip()
HAS_VERTEX = bool(GCP_PROJECT) and GOOGLE_CREDENTIALS.is_file()
# The Gemini image models are served from the "global" endpoint; Veo is regional.
GCP_IMAGE_LOCATION = os.getenv("GCP_IMAGE_LOCATION", "global").strip()
GCP_VIDEO_LOCATION = os.getenv("GCP_VIDEO_LOCATION", "us-central1").strip()

# Model ids as Vertex names them; verified against this project in Oct 2026.
VERTEX_IMAGE_MODELS = [
    "gemini-3.1-flash-image",  # Nano Banana 2: very accurate, ~10 s
    "gemini-3-pro-image",  # Nano Banana Pro: most accurate, ~50 s
]
VERTEX_VIDEO_MODELS = [
    "veo-3.1-fast-generate-001",  # ~40 s for 4 s of video, with sound
    "veo-3.1-generate-001",  # full quality, slower and pricier
    "veo-3.1-lite-generate-001",  # cheapest
]

# With a Google key the defaults are the Google models; otherwise they were
# chosen in Oct 2026 for prompt accuracy AND for having a free, running public
# Space, so they work even when the token cannot call the metered Inference API.
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL", VERTEX_IMAGE_MODELS[0] if HAS_VERTEX else "Tongyi-MAI/Z-Image-Turbo"
).strip()
VIDEO_MODEL = os.getenv(
    "VIDEO_MODEL", VERTEX_VIDEO_MODELS[0] if HAS_VERTEX else "MiniMaxAI/MiniMax-H3"
).strip()
TEXT_MODEL = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()

# Public Spaces that serve the same open-source models without a token.
# Each entry is tried in order; the first that answers wins. The Gradio call
# for each Space lives in image_tool.SPACE_ADAPTERS / video_tool.SPACE_ADAPTERS.
SPACE_MAP = {
    "Tongyi-MAI/Z-Image-Turbo": [
        "mrfakename/Z-Image-Turbo",
        "Tongyi-MAI/Z-Image-Turbo",
    ],
    "krea/Krea-2-Turbo": ["krea/Krea-2"],
    # The 9B Space is broken upstream (RuntimeError on every call, Oct 2026); the
    # 4B Space is healthy and ~11 s per image.
    "black-forest-labs/FLUX.2-klein-4B": ["black-forest-labs/FLUX.2-klein-4B"],
    "black-forest-labs/FLUX.1-schnell": [
        "black-forest-labs/FLUX.1-schnell",
        "multimodalart/FLUX.1-schnell",
    ],
}
VIDEO_SPACE_MAP = {
    "MiniMaxAI/MiniMax-H3": [
        "multimodalart/minimax-h3",  # full 28-step quality, with soundtrack
        "hugging-apps/minimax-h3-flashgen-4step",  # 4-step distilled, much faster
    ],
    "Lightricks/LTX-Video-0.9.7-distilled": ["Lightricks/ltx-video-distilled"],
}
# Models the Inference API only serves under a different task (image editing,
# image-to-video), so a text prompt can only go to their Space.
SPACE_ONLY = {"black-forest-labs/FLUX.2-klein-4B", "MiniMaxAI/MiniMax-H3"}

_BACKEND = os.getenv("BACKEND", "auto").strip().lower()
# "auto": use the Inference API when a token exists, otherwise fall back to the
# public Space so the app still produces real images out of the box.
BACKEND = _BACKEND if _BACKEND != "auto" else ("api" if HF_TOKEN else "space")

OUTPUT_DIR = ROOT / os.getenv("OUTPUT_DIR", "generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Image defaults: (steps, guidance scale) per model
# --------------------------------------------------------------------------- #
MODEL_DEFAULTS: Dict[str, Tuple[int, float]] = {
    # Google models pick their own steps and guidance; these are display-only.
    "gemini-3.1-flash-image": (1, 0.0),
    "gemini-3-pro-image": (1, 0.0),
    "Tongyi-MAI/Z-Image-Turbo": (9, 0.0),
    "krea/Krea-2-Turbo": (8, 0.0),
    "black-forest-labs/FLUX.2-klein-4B": (4, 1.0),
    "black-forest-labs/FLUX.2-klein-9B": (4, 1.0),
    "Qwen/Qwen-Image-2512": (30, 4.0),
    "tencent/HunyuanImage-3.0": (28, 7.0),
    "black-forest-labs/FLUX.1-dev": (28, 3.5),
    "black-forest-labs/FLUX.1-schnell": (4, 0.0),
}
# Models that only need a handful of steps (turbo / schnell distilled variants).
FAST_MODELS = ("schnell", "turbo", "lightning", "lcm", "klein", "distilled")


def default_steps(model_id: str = IMAGE_MODEL) -> int:
    if model_id in MODEL_DEFAULTS:
        return MODEL_DEFAULTS[model_id][0]
    return 4 if any(tag in model_id.lower() for tag in FAST_MODELS) else 28


def default_guidance(model_id: str = IMAGE_MODEL) -> float:
    if model_id in MODEL_DEFAULTS:
        return MODEL_DEFAULTS[model_id][1]
    return 0.0 if any(tag in model_id.lower() for tag in FAST_MODELS) else 7.0


def uses_negative_prompt(model_id: str = IMAGE_MODEL) -> bool:
    """Few-step models run without classifier-free guidance and ignore negatives."""
    return default_guidance(model_id) > 1.0


# --------------------------------------------------------------------------- #
# Video defaults
# --------------------------------------------------------------------------- #
# Clip length in seconds. MiniMax-H3 needs at least 2 s; longer clips take
# proportionally more of the free ZeroGPU quota.
VIDEO_DURATION = float(os.getenv("VIDEO_DURATION", "4"))
VIDEO_MIN_DURATION = 1.0
VIDEO_MAX_DURATION = 10.0
# (steps, guidance, fps) per model.
VIDEO_MODEL_DEFAULTS: Dict[str, Tuple[int, float, int]] = {
    "veo-3.1-fast-generate-001": (1, 0.0, 24),
    "veo-3.1-generate-001": (1, 0.0, 24),
    "veo-3.1-lite-generate-001": (1, 0.0, 24),
    "MiniMaxAI/MiniMax-H3": (28, 1.0, 24),
    "Wan-AI/Wan2.2-T2V-A14B": (30, 5.0, 16),
    "tencent/HunyuanVideo-1.5": (30, 6.0, 24),
    "Lightricks/LTX-Video-0.9.7-distilled": (8, 1.0, 30),
}
VIDEO_FPS = {"ltx": 30, "wan": 16, "hunyuan": 24, "minimax": 24, "mochi": 30, "cogvideo": 8}
# Default frame size. Multiples of 32, which LTX requires; the MiniMax Spaces
# snap it to their nearest canvas.
VIDEO_SIZE = (960, 544)


def video_fps(model_id: str = VIDEO_MODEL) -> int:
    if model_id in VIDEO_MODEL_DEFAULTS:
        return VIDEO_MODEL_DEFAULTS[model_id][2]
    lowered = model_id.lower()
    for tag, fps in VIDEO_FPS.items():
        if tag in lowered:
            return fps
    return 24


def default_video_steps(model_id: str = VIDEO_MODEL) -> int:
    if model_id in VIDEO_MODEL_DEFAULTS:
        return VIDEO_MODEL_DEFAULTS[model_id][0]
    lowered = model_id.lower()
    return 8 if ("distilled" in lowered or "turbo" in lowered) else 30


def default_video_guidance(model_id: str = VIDEO_MODEL) -> float:
    if model_id in VIDEO_MODEL_DEFAULTS:
        return VIDEO_MODEL_DEFAULTS[model_id][1]
    return 1.0 if "distilled" in model_id.lower() else 5.0


def quota_help(what: str) -> str:
    """Message for a ZeroGPU 'runs limit' / quota refusal from a public Space."""
    if HF_TOKEN:
        return (
            f"Today's free ZeroGPU quota for your Hugging Face account is used up, so the "
            f"public Space refused to render more {what}. It resets daily. Options: wait, "
            "subscribe to PRO for 40 min of ZeroGPU a day (https://huggingface.co/subscribe/pro), "
            "or enable \"Make calls to Inference Providers\" on your token "
            "(https://huggingface.co/settings/tokens) so the metered API can be used instead."
        )
    return (
        f"The free ZeroGPU quota for anonymous use is exhausted, so the public Space refused "
        f"to render more {what}. Add HUGGINGFACEHUB_API_TOKEN to your .env for a much larger "
        "daily quota (free: https://huggingface.co/settings/tokens), or try again later."
    )


# Shown when the Inference API refuses the token. Fine-grained tokens need the
# "Make calls to Inference Providers" permission; classic Read tokens have it.
PERMISSION_HELP = (
    "Your Hugging Face token cannot call Inference Providers (403), and {model} has no "
    "free public Space. Either pick a model marked 'free Space' or enable "
    "\"Make calls to Inference Providers\" on your token at "
    "https://huggingface.co/settings/tokens and restart the server."
)
