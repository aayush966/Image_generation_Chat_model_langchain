"""A LangChain tool that turns a text prompt into an image with an open-source
Hugging Face diffusion model.

Backends:
  * "api"   -> huggingface_hub.InferenceClient.text_to_image (Inference Providers)
  * "space" -> a public Hugging Face Space running the model on ZeroGPU (no token)
  * "local" -> diffusers pipeline running on this machine (needs torch + diffusers)
  * "demo"  -> a labelled placeholder, for exercising the UI offline

Google models (config.VERTEX_IMAGE_MODELS) always go to Vertex AI, whatever the
backend, since they are only served there.
"""

from __future__ import annotations

import io
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr

import config


class ImageRequest(BaseModel):
    """Arguments accepted by the image generation tool."""

    prompt: str = Field(description="A detailed description of the image to create.")
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, watermark, text artifacts",
        description="What the image should avoid.",
    )
    width: int = Field(default=1024, description="Image width in pixels.")
    height: int = Field(default=1024, description="Image height in pixels.")
    steps: Optional[int] = Field(default=None, description="Denoising steps.")
    guidance_scale: Optional[float] = Field(
        default=None, description="How strictly to follow the prompt."
    )
    seed: Optional[int] = Field(default=None, description="Seed for reproducible output.")
    model: Optional[str] = Field(
        default=None, description="Model repo id; falls back to the configured default."
    )


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit] or "image").rstrip("-")


def _snap(value: int, step: int = 16, low: int = 256, high: int = 2048) -> int:
    return int(min(high, max(low, round(value / step) * step)))


def _steps(req: ImageRequest, model_id: str) -> int:
    return int(req.steps or config.default_steps(model_id))


def _guidance(req: ImageRequest, model_id: str) -> float:
    return float(
        req.guidance_scale
        if req.guidance_scale is not None
        else config.default_guidance(model_id)
    )


def is_permission_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "403" in message or "402" in message or "permission" in message


_vertex_clients: Dict[str, Any] = {}


def vertex_client(location: str) -> Any:
    """A google-genai client on Vertex AI, one per region, authenticated by the
    service-account key in GOOGLE_APPLICATION_CREDENTIALS."""
    if not config.HAS_VERTEX:
        raise RuntimeError(
            f"Google models need a service-account key: put it at {config.GOOGLE_CREDENTIALS} "
            "or set GOOGLE_APPLICATION_CREDENTIALS in .env."
        )
    if location not in _vertex_clients:
        from google import genai

        _vertex_clients[location] = genai.Client(
            vertexai=True, project=config.GCP_PROJECT, location=location
        )
    return _vertex_clients[location]


def vertex_error(exc: BaseException, what: str) -> RuntimeError:
    """Turn a Vertex failure into a message the chat UI can show as-is."""
    message = str(exc)
    if "429" in message or "RESOURCE_EXHAUSTED" in message:
        return RuntimeError(f"Google Vertex AI rate limit or quota hit for {what}; wait a minute and retry.")
    if "403" in message or "PERMISSION_DENIED" in message:
        return RuntimeError(
            f"The Google key cannot call {what} (403). Give the service account the "
            "'Vertex AI User' role and enable the Vertex AI API. Details: " + message[:300]
        )
    return RuntimeError(f"Google Vertex AI failed for {what}: {type(exc).__name__}: {message[:400]}")


# --------------------------------------------------------------------------- #
# Space adapters: one Gradio call per public Space
# --------------------------------------------------------------------------- #
def closest_label(labels: Iterable[str], width: int, height: int) -> str:
    """Pick the 'WxH ...' label whose aspect ratio, then pixel count, is nearest."""
    want_ratio = math.log(width / height)
    want_pixels = width * height
    parsed: List[Tuple[str, int, int]] = []
    for label in labels:
        match = re.match(r"\s*(\d+)\s*[x×]\s*(\d+)", label)
        if match:
            parsed.append((label, int(match.group(1)), int(match.group(2))))
    if not parsed:
        raise ValueError("no WxH labels to choose from")
    best_ratio = min(abs(math.log(w / h) - want_ratio) for _, w, h in parsed)
    candidates = [p for p in parsed if abs(math.log(p[1] / p[2]) - want_ratio) <= best_ratio + 0.05]
    return min(candidates, key=lambda p: abs(p[1] * p[2] - want_pixels))[0]


def result_path(result: Any) -> str:
    """Pull a file path out of whatever shape a Space returned."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):  # a Gallery
        result = result[0]
    if isinstance(result, dict):
        inner = result.get("image") or result.get("video") or result.get("path") or result.get("url")
        if isinstance(inner, dict):
            inner = inner.get("path") or inner.get("url")
        result = inner
    if not result:
        raise RuntimeError("the Space returned no file")
    return str(result)


def _flux_schnell(client: Any, req: ImageRequest, model_id: str) -> Any:
    return client.predict(
        prompt=req.prompt,
        # These must be ints: the Space's own code indexes with them.
        seed=int(req.seed or 0),
        randomize_seed=req.seed is None,
        width=int(req.width),
        height=int(req.height),
        # The schnell Spaces cap the step slider well below the API's limit.
        num_inference_steps=int(min(req.steps or 4, 8)),
        api_name="/infer",
    )


def _z_image_mrfakename(client: Any, req: ImageRequest, model_id: str) -> Any:
    return client.predict(
        prompt=req.prompt,
        height=_snap(req.height),
        width=_snap(req.width),
        num_inference_steps=int(min(_steps(req, model_id), 50)),
        seed=int(req.seed or 0),
        randomize_seed=req.seed is None,
        api_name="/generate_image",
    )


Z_IMAGE_RESOLUTIONS = [
    "1024x1024 ( 1:1 )", "1152x896 ( 9:7 )", "896x1152 ( 7:9 )", "1152x864 ( 4:3 )",
    "864x1152 ( 3:4 )", "1248x832 ( 3:2 )", "832x1248 ( 2:3 )", "1280x720 ( 16:9 )",
    "720x1280 ( 9:16 )", "1344x576 ( 21:9 )", "576x1344 ( 9:21 )", "1280x1280 ( 1:1 )",
    "1440x1120 ( 9:7 )", "1120x1440 ( 7:9 )", "1472x1104 ( 4:3 )", "1104x1472 ( 3:4 )",
    "1536x1024 ( 3:2 )", "1024x1536 ( 2:3 )", "1536x864 ( 16:9 )", "864x1536 ( 9:16 )",
    "1680x720 ( 21:9 )", "720x1680 ( 9:21 )", "1536x1536 ( 1:1 )",
]


def _z_image_official(client: Any, req: ImageRequest, model_id: str) -> Any:
    return client.predict(
        prompt=req.prompt,
        resolution=closest_label(Z_IMAGE_RESOLUTIONS, req.width, req.height),
        seed=int(req.seed or 0),
        steps=int(min(_steps(req, model_id), 50)),
        shift=3.0,
        random_seed=req.seed is None,
        gallery_images=[],
        api_name="/generate",
    )


def _krea2(client: Any, req: ImageRequest, model_id: str) -> Any:
    guidance = _guidance(req, model_id)
    return client.predict(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt if guidance > 0 else "",
        model="Turbo",
        steps=int(min(_steps(req, model_id), 50)),
        guidance=float(min(guidance, 10.0)),
        width=_snap(req.width, 16, 512, 2048),
        height=_snap(req.height, 16, 512, 2048),
        seed=int(req.seed or 0),  # Spaces feed this to a torch Generator: must be an int
        randomize=req.seed is None,
        api_name="/generate",
    )


def _flux2_klein(client: Any, req: ImageRequest, model_id: str) -> Any:
    # The 4B Space names the endpoint /infer, the 9B one /generate.
    api_name = "/infer" if "4B" in model_id else "/generate"
    steps = _steps(req, model_id)
    return client.predict(
        prompt=req.prompt,
        input_images=[],  # empty = plain text-to-image
        mode_choice="Distilled (4 steps)" if steps <= 8 else "Base (50 steps)",
        seed=int(req.seed or 0),  # Spaces feed this to a torch Generator: must be an int
        randomize_seed=req.seed is None,
        width=_snap(req.width),
        height=_snap(req.height),
        num_inference_steps=int(min(steps, 50)),
        guidance_scale=_guidance(req, model_id),
        prompt_upsampling=False,
        api_name=api_name,
    )


QWEN_ASPECTS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


def _qwen_image(client: Any, req: ImageRequest, model_id: str) -> Any:
    # The Space picks the size from an aspect ratio, so turn w/h into the nearest.
    labels = [f"{a.split(':')[0]}x{a.split(':')[1]} {a}" for a in QWEN_ASPECTS]
    aspect = closest_label(labels, req.width, req.height).split(" ")[1]
    return client.predict(
        prompt=req.prompt,
        seed=int(req.seed or 0),  # Spaces feed this to a torch Generator: must be an int
        randomize_seed=req.seed is None,
        aspect_ratio=aspect,
        guidance_scale=_guidance(req, model_id),
        num_inference_steps=int(min(_steps(req, model_id), 50)),
        # Our LangChain enhancer already expanded the prompt.
        prompt_enhance=False,
        api_name="/infer",
    )


# Aspect ratios the Gemini image models accept.
VERTEX_ASPECTS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def nano_banana(
    model_id: str,
    prompt: str,
    width: int,
    height: int,
    seed: Optional[int] = None,
    image: Optional[Image.Image] = None,
) -> Image.Image:
    """One call to a Nano Banana (Gemini image) model on Vertex AI.

    With `image`, the model edits/re-renders that picture following `prompt`
    instead of drawing from scratch.
    """
    from google.genai import types

    labels = [f"{a.replace(':', 'x')} {a}" for a in VERTEX_ASPECTS]
    aspect = closest_label(labels, width, height).split(" ")[1]
    config_kwargs: Dict[str, Any] = {
        "response_modalities": ["IMAGE"],
        "image_config": types.ImageConfig(
            aspect_ratio=aspect,
            image_size="2K" if max(width, height) > 1536 else "1K",
        ),
    }
    if seed is not None:
        config_kwargs["seed"] = int(seed)

    contents: Any = prompt
    if image is not None:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        contents = [types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"), prompt]

    try:
        response = vertex_client(config.GCP_IMAGE_LOCATION).models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as exc:
        raise vertex_error(exc, model_id) from exc

    candidate = (response.candidates or [None])[0]
    for part in (candidate.content.parts or []) if candidate and candidate.content else []:
        if part.inline_data and part.inline_data.data:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    reason = getattr(candidate, "finish_reason", None) or getattr(
        response.prompt_feedback, "block_reason", None
    )
    raise RuntimeError(
        f"{model_id} returned no image (reason: {reason}). Google refuses some prompts, "
        "e.g. real public figures; try rephrasing."
    )

SPACE_ADAPTERS: Dict[str, Callable[[Any, ImageRequest, str], Any]] = {
    "black-forest-labs/FLUX.1-schnell": _flux_schnell,
    "multimodalart/FLUX.1-schnell": _flux_schnell,
    "mrfakename/Z-Image-Turbo": _z_image_mrfakename,
    "Tongyi-MAI/Z-Image-Turbo": _z_image_official,
    "krea/Krea-2": _krea2,
    "black-forest-labs/FLUX.2-klein-4B": _flux2_klein,
    "black-forest-labs/FLUX.2-klein-9B": _flux2_klein,
    # Kept for when the official Space is fixed; it currently raises a NameError.
    "Qwen/Qwen-Image": _qwen_image,
}


class HuggingFaceImageTool(BaseTool):
    """LangChain tool wrapper around a Hugging Face text-to-image model."""

    name: str = "generate_image"
    description: str = (
        "Generate an image from a text description using an open-source "
        "Hugging Face diffusion model. Input should be a rich, visual prompt. "
        "Returns the file path of the saved PNG."
    )
    args_schema: Type[BaseModel] = ImageRequest

    model_id: str = config.IMAGE_MODEL
    backend: str = config.BACKEND
    output_dir: Path = config.OUTPUT_DIR

    # Clients and pipelines are cached per model, so concurrent requests asking
    # for different models never clobber each other.
    _clients: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _pipelines: Dict[str, Any] = PrivateAttr(default_factory=dict)
    # Models the Inference API has refused for this token (missing permission or
    # no credits). Remembered so we stop paying for a doomed round trip.
    _api_blocked: set = PrivateAttr(default_factory=set)

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _api_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        from huggingface_hub import InferenceClient

        if model_id not in self._clients:
            if not config.HF_TOKEN:
                raise RuntimeError(
                    "No Hugging Face token found. Put HUGGINGFACEHUB_API_TOKEN in your .env "
                    "(free read token: https://huggingface.co/settings/tokens)."
                )
            self._clients[model_id] = InferenceClient(
                model=model_id, token=config.HF_TOKEN, timeout=180
            )
        client = self._clients[model_id]

        kwargs = {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "num_inference_steps": _steps(req, model_id),
            "guidance_scale": _guidance(req, model_id),
        }
        # Few-step turbo models run without guidance and ignore negatives.
        if req.negative_prompt and config.uses_negative_prompt(model_id):
            kwargs["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            kwargs["seed"] = req.seed

        result = client.text_to_image(**kwargs)
        if isinstance(result, (bytes, bytearray)):
            return Image.open(io.BytesIO(result))
        return result

    def _vertex_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        """Google's Nano Banana (Gemini image) models on Vertex AI."""
        return nano_banana(model_id, req.prompt, req.width, req.height, req.seed)

    def _space_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        """Call a public Hugging Face Space - the open-source model runs on HF's
        hardware and no token is required."""
        from gradio_client import Client

        spaces = config.SPACE_MAP.get(model_id)
        if not spaces:
            raise RuntimeError(
                f"{model_id} has no public Space, so it needs a Hugging Face token that can "
                "call Inference Providers. Either pick a model marked 'free Space' or fix "
                "the token permission at https://huggingface.co/settings/tokens."
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
                return Image.open(result_path(result)).convert("RGB")
            except Exception as exc:  # try the next mirror
                self._clients.pop(space, None)
                errors.append(f"{space}: {type(exc).__name__}: {exc}".strip(": "))

        if any("quota" in error.lower() or "runs limit" in error.lower() for error in errors):
            raise RuntimeError(config.quota_help("images"))
        raise RuntimeError(
            "Every public Space for this model failed (they are free and sometimes "
            "queued or asleep). Details - " + " | ".join(errors)
        )

    def _demo_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        """No model, no token: a deterministic placeholder so the UI can be exercised."""
        from PIL import ImageDraw

        seed = req.seed if req.seed is not None else sum(map(ord, req.prompt))
        top = ((seed * 37) % 200 + 30, (seed * 61) % 200 + 30, (seed * 89) % 200 + 30)
        bottom = ((seed * 13) % 120, (seed * 29) % 120, (seed * 53) % 120)

        image = Image.new("RGB", (req.width, req.height))
        draw = ImageDraw.Draw(image)
        for y in range(req.height):
            ratio = y / max(req.height - 1, 1)
            draw.line(
                [(0, y), (req.width, y)],
                fill=tuple(int(t + (b - t) * ratio) for t, b in zip(top, bottom)),
            )

        words, lines, line = req.prompt.split(), [], ""
        for word in words:
            if len(line) + len(word) > 34:
                lines.append(line.strip())
                line = ""
            line += word + " "
        lines.append(line.strip())

        draw.text((28, 24), "DEMO MODE - no model called", fill=(255, 255, 255))
        for index, text in enumerate(lines[:8]):
            draw.text((28, 56 + index * 18), text, fill=(240, 240, 240))
        return image

    def _local_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        import torch
        from diffusers import AutoPipelineForText2Image

        if model_id not in self._pipelines:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=dtype, token=config.HF_TOKEN or None
            )
            self._pipelines[model_id] = pipe.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        pipeline = self._pipelines[model_id]

        generator = None
        if req.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(req.seed)

        kwargs = {
            "prompt": req.prompt,
            "width": req.width,
            "height": req.height,
            "num_inference_steps": _steps(req, model_id),
            "guidance_scale": _guidance(req, model_id),
            "generator": generator,
        }
        if req.negative_prompt and config.uses_negative_prompt(model_id):
            kwargs["negative_prompt"] = req.negative_prompt

        return pipeline(**kwargs).images[0]

    # ------------------------------------------------------------------ #
    # BaseTool interface
    # ------------------------------------------------------------------ #
    def _run(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, watermark, text artifacts",
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        req = ImageRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )
        # Resolved per call - never stored on the shared tool, so parallel
        # requests for different models cannot clobber each other.
        model_id = req.model or self.model_id

        backends = {
            "local": self._local_image,
            "demo": self._demo_image,
            "space": self._space_image,
            "api": self._api_image,
            "vertex": self._vertex_image,
        }
        backend = self.backend
        # Known to be refused by the API for this token, or only served by the
        # API as an image-editing task: go straight to the Space.
        if backend == "api" and (model_id in self._api_blocked or model_id in config.SPACE_ONLY):
            backend = "space"
        if model_id in config.VERTEX_IMAGE_MODELS and backend != "demo":
            backend = "vertex"

        try:
            image = backends.get(backend, self._api_image)(req, model_id)
        except Exception as exc:
            # The Inference API needs the "Inference Providers" token permission and
            # is metered; the same model on a public Space has its own quota.
            blocked = is_permission_error(exc)
            if blocked:
                self._api_blocked.add(model_id)
            if backend != "api" or model_id not in config.SPACE_MAP:
                if blocked and backend == "api":
                    raise RuntimeError(config.PERMISSION_HELP.format(model=model_id)) from exc
                raise
            image = self._space_image(req, model_id)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{int(time.time())}-{_slugify(req.prompt)}.png"
        image.save(path)
        return str(path)


image_tool = HuggingFaceImageTool()
