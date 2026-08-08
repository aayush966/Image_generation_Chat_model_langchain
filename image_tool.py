"""A LangChain tool that turns a text prompt into an image with an open-source
Hugging Face diffusion model.

Two backends:
  * "api"   -> huggingface_hub.InferenceClient.text_to_image (nothing to install locally)
  * "local" -> diffusers pipeline running on this machine (needs torch + diffusers)
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type

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
            "num_inference_steps": req.steps or config.default_steps(model_id),
            "guidance_scale": (
                req.guidance_scale
                if req.guidance_scale is not None
                else config.default_guidance(model_id)
            ),
        }
        # FLUX ignores negative prompts; SD-family models use them.
        if req.negative_prompt and "flux" not in model_id.lower():
            kwargs["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            kwargs["seed"] = req.seed

        result = client.text_to_image(**kwargs)
        if isinstance(result, (bytes, bytearray)):
            return Image.open(io.BytesIO(result))
        return result

    def _space_image(self, req: ImageRequest, model_id: str) -> Image.Image:
        """Call a public Hugging Face Space - the open-source model runs on HF's
        hardware and no token is required."""
        from gradio_client import Client

        spaces = config.SPACE_MAP.get(model_id)
        if not spaces:
            raise RuntimeError(
                f"{model_id} has no public Space, so it needs a Hugging Face token. "
                "Either switch the model to FLUX.1-schnell (works without a token) "
                "or add HUGGINGFACEHUB_API_TOKEN to your .env file."
            )

        errors = []
        for space in spaces:
            try:
                if space not in self._clients:
                    # A token is optional here, but it raises the ZeroGPU quota a lot.
                    self._clients[space] = Client(
                        space, token=config.HF_TOKEN or None, verbose=False
                    )
                client = self._clients[space]
                result = client.predict(
                    prompt=req.prompt,
                    # These must be ints: the Space's own code indexes with them and
                    # raises a server-side TypeError on floats, even though its
                    # API schema advertises "float".
                    seed=int(req.seed or 0),
                    randomize_seed=req.seed is None,
                    width=int(req.width),
                    height=int(req.height),
                    # The schnell Spaces cap the step slider well below the API's limit.
                    num_inference_steps=int(min(req.steps or 4, 8)),
                    api_name="/infer",
                )
                path = result[0] if isinstance(result, (list, tuple)) else result
                if isinstance(path, dict):
                    path = path.get("path") or path.get("url")
                return Image.open(path).convert("RGB")
            except Exception as exc:  # try the next mirror
                self._clients.pop(space, None)
                errors.append(f"{space}: {type(exc).__name__}: {exc}".strip(": "))

        if any("quota" in error.lower() for error in errors):
            raise RuntimeError(
                "The free Hugging Face Space quota for this computer is used up - "
                "anonymous users only get a few images per hour. Add "
                "HUGGINGFACEHUB_API_TOKEN to your .env file for a much larger quota "
                "(free, takes two minutes: https://huggingface.co/settings/tokens), "
                "or wait a while and try again."
            )
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
            "num_inference_steps": req.steps or config.default_steps(model_id),
            "guidance_scale": (
                req.guidance_scale
                if req.guidance_scale is not None
                else config.default_guidance(model_id)
            ),
            "generator": generator,
        }
        if req.negative_prompt and "flux" not in model_id.lower():
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
        }
        backend = self.backend
        # Already known to be refused by the Inference API - go straight to the Space.
        if backend == "api" and model_id in self._api_blocked:
            backend = "space"

        try:
            image = backends.get(backend, self._api_image)(req, model_id)
        except Exception as exc:
            # The Inference API needs the "Inference Providers" token permission and
            # is metered; the same model on a public Space has its own quota.
            if backend != "api" or model_id not in config.SPACE_MAP:
                raise
            if "403" in str(exc) or "permission" in str(exc).lower():
                self._api_blocked.add(model_id)
            image = self._space_image(req, model_id)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{int(time.time())}-{_slugify(req.prompt)}.png"
        image.save(path)
        return str(path)


image_tool = HuggingFaceImageTool()
