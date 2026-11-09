"""A LangChain tool that turns a text prompt into a short video clip with an
open-source Hugging Face text-to-video model.

Backends, same shape as image_tool.py:
  * "api"   -> huggingface_hub.InferenceClient.text_to_video (Inference Providers)
  * "space" -> a public Hugging Face Space running the model on ZeroGPU (no token)
  * "local" -> diffusers pipeline on this machine (needs torch + diffusers + imageio)
  * "demo"  -> an animated placeholder, for exercising the UI offline

Google Veo models (config.VERTEX_VIDEO_MODELS) always go to Vertex AI.
"""

from __future__ import annotations

import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

import config
from image_tool import closest_label, is_permission_error, result_path, vertex_client, vertex_error

DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark"


class VideoRequest(BaseModel):
    """Arguments accepted by the video generation tool."""

    prompt: str = Field(description="A detailed description of the clip: subject, motion, camera.")
    negative_prompt: str = Field(default=DEFAULT_NEGATIVE, description="What the clip should avoid.")
    width: int = Field(default=config.VIDEO_SIZE[0], description="Frame width in pixels.")
    height: int = Field(default=config.VIDEO_SIZE[1], description="Frame height in pixels.")
    duration: float = Field(
        default=config.VIDEO_DURATION, description="Clip length in seconds (1 - 10)."
    )
    steps: Optional[int] = Field(default=None, description="Denoising steps.")
    guidance_scale: Optional[float] = Field(default=None, description="Prompt adherence.")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible output.")
    model: Optional[str] = Field(default=None, description="Model repo id; defaults to VIDEO_MODEL.")


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit] or "video").rstrip("-")


def _snap32(value: int, low: int = 256, high: int = 1280) -> int:
    """Round to the nearest multiple of 32 inside [low, high]."""
    return int(min(high, max(low, round(value / 32) * 32)))


def _clamp(seconds: float, low: float = config.VIDEO_MIN_DURATION, high: float = config.VIDEO_MAX_DURATION) -> float:
    return float(min(high, max(low, seconds)))


def _num_frames(seconds: float, fps: int) -> int:
    """Diffusion video models want 8k+1 frames; convert seconds to the nearest."""
    frames = max(9, round(seconds * fps))
    return int(round((frames - 1) / 8) * 8 + 1)


def _steps(req: VideoRequest, model_id: str) -> int:
    return int(req.steps or config.default_video_steps(model_id))


def _guidance(req: VideoRequest, model_id: str) -> float:
    return float(
        req.guidance_scale
        if req.guidance_scale is not None
        else config.default_video_guidance(model_id)
    )


# --------------------------------------------------------------------------- #
# Space adapters: one Gradio call per public Space
# --------------------------------------------------------------------------- #
def _ltx_distilled(client: Any, req: VideoRequest, model_id: str) -> Any:
    return client.predict(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or DEFAULT_NEGATIVE,
        input_image_filepath=None,
        input_video_filepath=None,
        height_ui=_snap32(req.height),
        width_ui=_snap32(req.width),
        mode="text-to-video",
        duration_ui=_clamp(req.duration, 0.5, 8.5),
        ui_frames_to_use=9,
        seed_ui=int(req.seed or 0),
        randomize_seed=req.seed is None,
        ui_guidance_scale=_guidance(req, model_id),
        improve_texture_flag=True,
        api_name="/text_to_video",
    )


# The canvas labels both MiniMax-H3 Spaces accept. "fast" tiers render in a
# fraction of the time of "full" ones; closest_label picks by aspect, then size.
MINIMAX_CANVASES = [
    "960x544 · 16:9 fast", "1024x576 · 16:9 fast", "1152x640 · 16:9", "1280x704 · 16:9",
    "1344x768 · 16:9 full", "544x960 · 9:16 fast", "640x1152 · 9:16", "768x1344 · 9:16 full",
    "544x544 · 1:1 fast", "768x768 · 1:1 full", "768x576 · 4:3 fast", "1024x768 · 4:3 full",
    "576x768 · 3:4 fast", "768x1024 · 3:4 full", "1152x512 · 21:9 fast", "1536x672 · 21:9 full",
]


def _minimax_args(req: VideoRequest) -> Dict[str, Any]:
    return {
        "prompt": req.prompt,
        "image_path": None,
        "last_image_path": None,
        "canvas": closest_label(MINIMAX_CANVASES, req.width, req.height),
        # The Spaces' slider runs 2-14 s in whole seconds.
        "duration": float(round(_clamp(req.duration, 2.0, 14.0))),
        "seed": int(req.seed if req.seed is not None else random.randint(0, 2**31 - 1)),
    }


def _minimax_full(client: Any, req: VideoRequest, model_id: str) -> Any:
    """multimodalart/minimax-h3: the full model, 10-40 steps, with soundtrack."""
    return client.predict(
        **_minimax_args(req),
        steps=int(min(max(_steps(req, model_id), 10), 40)),
        upsample=False,  # our LangChain enhancer already expanded the prompt
        api_name="/generate",
    )


def _minimax_flash(client: Any, req: VideoRequest, model_id: str) -> Any:
    """hugging-apps/minimax-h3-flashgen-4step: distilled 4-step variant, much faster."""
    return client.predict(
        **_minimax_args(req),
        enhance_prompt=False,
        api_name="/generate",
    )


# Seconds to wait after each "high load" refusal from Veo; None = last attempt.
VEO_RETRY_WAITS = (15, 30, None)


def _veo_busy(message: str) -> bool:
    lowered = message.lower()
    return "high load" in lowered or "'code': 8" in lowered or "resource_exhausted" in lowered


def veo(model_id: str, prompt: str, config_kwargs: Dict[str, Any], image: Any = None) -> bytes:
    """Run one Veo job on Vertex AI and return the MP4 bytes.

    `image` (a google.genai types.Image) makes it image-to-video: the clip starts
    from that picture. Veo often answers "high load" (code 8) at busy times; that
    clears within a minute or so, so retry a few times before giving up.
    """
    from google.genai import types

    client = vertex_client(config.GCP_VIDEO_LOCATION)
    operation = None
    for wait in VEO_RETRY_WAITS:
        try:
            operation = client.models.generate_videos(
                model=model_id,
                prompt=prompt,
                image=image,
                config=types.GenerateVideosConfig(**config_kwargs),
            )
            deadline = time.time() + 600
            while not operation.done:
                if time.time() > deadline:
                    raise TimeoutError("Veo did not finish within 10 minutes")
                time.sleep(5)
                operation = client.operations.get(operation)
        except Exception as exc:
            if _veo_busy(str(exc)) and wait is not None:
                time.sleep(wait)
                continue
            raise vertex_error(exc, model_id) from exc
        if operation.error and _veo_busy(str(operation.error)) and wait is not None:
            time.sleep(wait)
            continue
        break

    if operation.error:
        if _veo_busy(str(operation.error)):
            raise RuntimeError(
                f"Google's {model_id} is overloaded right now (tried {len(VEO_RETRY_WAITS)} "
                "times). Try again in a few minutes, or pick another Veo model."
            )
        raise RuntimeError(f"{model_id} failed: {operation.error}")
    response = operation.response
    videos = response.generated_videos if response else None
    if not videos:
        reasons = getattr(response, "rai_media_filtered_reasons", None)
        raise RuntimeError(
            "Google blocked this video under its safety rules (you are not charged). "
            "Veo refuses real people such as politicians and celebrities; describe a "
            "fictional character instead." + (f" Google said: {reasons}" if reasons else "")
        )
    return videos[0].video.video_bytes


SPACE_ADAPTERS: Dict[str, Callable[[Any, VideoRequest, str], Any]] = {
    "Lightricks/ltx-video-distilled": _ltx_distilled,
    "multimodalart/minimax-h3": _minimax_full,
    "hugging-apps/minimax-h3-flashgen-4step": _minimax_flash,
}


class HuggingFaceVideoTool(BaseTool):
    """LangChain tool wrapper around a Hugging Face text-to-video model."""

    name: str = "generate_video"
    description: str = (
        "Generate a short video clip from a text description using an open-source "
        "Hugging Face text-to-video model. Input should be a vivid prompt that also "
        "describes motion and camera movement. Returns the file path of the saved MP4."
    )
    args_schema: Type[BaseModel] = VideoRequest

    model_id: str = config.VIDEO_MODEL
    backend: str = config.BACKEND
    output_dir: Path = config.OUTPUT_DIR

    _clients: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _pipelines: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _api_blocked: set = PrivateAttr(default_factory=set)

    # ------------------------------------------------------------------ #
    # Backends - each returns the path of a finished file inside output_dir
    # ------------------------------------------------------------------ #
    def _target(self, req: VideoRequest, suffix: str = ".mp4") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{int(time.time())}-{_slugify(req.prompt)}{suffix}"

    def _api_video(self, req: VideoRequest, model_id: str) -> Path:
        from huggingface_hub import InferenceClient

        if model_id not in self._clients:
            if not config.HF_TOKEN:
                raise RuntimeError(
                    "No Hugging Face token found. Put HUGGINGFACEHUB_API_TOKEN in your .env "
                    "(free read token: https://huggingface.co/settings/tokens)."
                )
            # Video takes a while; the default 60 s client timeout is far too short.
            self._clients[model_id] = InferenceClient(
                model=model_id, token=config.HF_TOKEN, timeout=600
            )
        client = self._clients[model_id]

        kwargs: Dict[str, Any] = {
            "prompt": req.prompt,
            "num_frames": _num_frames(_clamp(req.duration), config.video_fps(model_id)),
            "num_inference_steps": _steps(req, model_id),
            "guidance_scale": _guidance(req, model_id),
        }
        if req.negative_prompt:
            kwargs["negative_prompt"] = [req.negative_prompt]
        if req.seed is not None:
            kwargs["seed"] = req.seed

        data = client.text_to_video(**kwargs)
        path = self._target(req)
        path.write_bytes(bytes(data))
        return path

    def _vertex_video(self, req: VideoRequest, model_id: str) -> Path:
        """Google Veo on Vertex AI. Long-running: see veo()."""
        # Veo renders 4, 6 or 8 second clips, landscape or portrait only.
        duration = min((4, 6, 8), key=lambda s: abs(s - req.duration))
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": "9:16" if req.height > req.width else "16:9",
            "duration_seconds": duration,
            "number_of_videos": 1,
        }
        if req.negative_prompt:
            config_kwargs["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            config_kwargs["seed"] = int(req.seed)

        path = self._target(req)
        path.write_bytes(veo(model_id, req.prompt, config_kwargs))
        return path

    def _space_video(self, req: VideoRequest, model_id: str) -> Path:
        """Call a public Hugging Face Space - the model runs on HF's ZeroGPU
        hardware and no token is required."""
        from gradio_client import Client

        spaces = config.VIDEO_SPACE_MAP.get(model_id)
        if not spaces:
            raise RuntimeError(
                f"{model_id} has no public Space, so it needs a Hugging Face token that can "
                "call Inference Providers. Either pick a video model marked 'free Space' or "
                "fix the token permission at https://huggingface.co/settings/tokens."
            )

        errors = []
        for space in spaces:
            adapter = SPACE_ADAPTERS.get(space)
            if adapter is None:
                errors.append(f"{space}: no adapter for this Space")
                continue
            try:
                if space not in self._clients:
                    # A token is optional here, but it raises the ZeroGPU quota a lot.
                    self._clients[space] = Client(
                        space, token=config.HF_TOKEN or None, verbose=False
                    )
                result = adapter(self._clients[space], req, model_id)
                source = Path(result_path(result))
                path = self._target(req, source.suffix or ".mp4")
                shutil.copyfile(source, path)
                return path
            except Exception as exc:  # try the next mirror
                self._clients.pop(space, None)
                errors.append(f"{space}: {type(exc).__name__}: {exc}".strip(": "))

        if any("quota" in error.lower() or "runs limit" in error.lower() for error in errors):
            raise RuntimeError(config.quota_help("video clips"))
        raise RuntimeError(
            "Every public Space for this video model failed (they are free and "
            "sometimes queued or asleep). Details - " + " | ".join(errors)
        )

    def _demo_video(self, req: VideoRequest, model_id: str) -> Path:
        """No model, no token: an animated GIF placeholder so the UI can be exercised."""
        from PIL import Image, ImageDraw

        width, height = _snap32(req.width), _snap32(req.height)
        seed = req.seed if req.seed is not None else sum(map(ord, req.prompt))
        top = ((seed * 37) % 200 + 30, (seed * 61) % 200 + 30, (seed * 89) % 200 + 30)
        bottom = ((seed * 13) % 120, (seed * 29) % 120, (seed * 53) % 120)
        fps = 10
        total = max(8, int(_clamp(req.duration) * fps))

        frames = []
        for index in range(total):
            frame = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(frame)
            for y in range(height):
                ratio = y / max(height - 1, 1)
                draw.line(
                    [(0, y), (width, y)],
                    fill=tuple(int(t + (b - t) * ratio) for t, b in zip(top, bottom)),
                )
            # Something that visibly moves, so a stuck player is obvious.
            x = int((index / total) * (width - 60)) + 20
            draw.ellipse([x, height // 2 - 20, x + 40, height // 2 + 20], fill=(255, 255, 255))
            draw.text((24, 20), "DEMO MODE - no model called", fill=(255, 255, 255))
            draw.text((24, 40), req.prompt[:60], fill=(240, 240, 240))
            frames.append(frame)

        path = self._target(req, ".gif")
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
        )
        return path

    def _local_video(self, req: VideoRequest, model_id: str) -> Path:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video

        if model_id not in self._pipelines:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            pipe = DiffusionPipeline.from_pretrained(
                model_id, torch_dtype=dtype, token=config.HF_TOKEN or None
            )
            self._pipelines[model_id] = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = self._pipelines[model_id]

        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(req.seed)

        fps = config.video_fps(model_id)
        frames = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=_snap32(req.width),
            height=_snap32(req.height),
            num_frames=_num_frames(_clamp(req.duration), fps),
            num_inference_steps=_steps(req, model_id),
            guidance_scale=_guidance(req, model_id),
            generator=generator,
        ).frames[0]

        path = self._target(req)
        export_to_video(frames, str(path), fps=fps)
        return path

    # ------------------------------------------------------------------ #
    # BaseTool interface
    # ------------------------------------------------------------------ #
    def _run(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE,
        width: int = config.VIDEO_SIZE[0],
        height: int = config.VIDEO_SIZE[1],
        duration: float = config.VIDEO_DURATION,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        req = VideoRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            duration=duration,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )
        # Resolved per call - never stored on the shared tool.
        model_id = req.model or self.model_id

        backends = {
            "local": self._local_video,
            "demo": self._demo_video,
            "space": self._space_video,
            "api": self._api_video,
            "vertex": self._vertex_video,
        }
        backend = self.backend
        # Known to be refused by the API for this token, or only served by the
        # API as image-to-video: go straight to the Space.
        if backend == "api" and (model_id in self._api_blocked or model_id in config.SPACE_ONLY):
            backend = "space"
        if model_id in config.VERTEX_VIDEO_MODELS and backend != "demo":
            backend = "vertex"

        try:
            path = backends.get(backend, self._api_video)(req, model_id)
        except Exception as exc:
            # The Inference API is metered and needs the "Inference Providers"
            # permission; the public Space has its own, separate quota.
            blocked = is_permission_error(exc)
            if blocked:
                self._api_blocked.add(model_id)
            if backend != "api" or model_id not in config.VIDEO_SPACE_MAP:
                if blocked and backend == "api":
                    raise RuntimeError(config.PERMISSION_HELP.format(model=model_id)) from exc
                raise
            path = self._space_video(req, model_id)

        return str(path)


video_tool = HuggingFaceVideoTool()
