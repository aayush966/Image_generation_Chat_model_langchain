"""3D Studio: turn an uploaded photo into new viewpoints, an orbit clip, or a
rotatable 3D model.

Three tools, all starting from one source photo in OUTPUT_DIR:

  * new_view()    -> Nano Banana re-renders the scene from another camera angle
                     (and can turn a chosen person toward the camera).
  * orbit_video() -> Veo animates a camera move around the scene (image-to-video).
  * to_mesh()     -> Hunyuan3D builds a textured GLB mesh on a free HF Space.

A single photo does not contain what is hidden in it. Every tool here *invents*
the parts the camera never saw (the back of a head, a turned-away face) - they
look plausible but are not a recovery of the real thing. The UI says so.
"""

from __future__ import annotations

import io
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps

import config
from image_tool import nano_banana, result_path

MAX_SIDE = 2048  # uploads are downscaled to this before anything else sees them
UPLOAD_PREFIX = "upload"

# Camera presets for new_view(): id -> how to describe the move to the model.
ANGLES: Dict[str, str] = {
    "left": "rotated about 45 degrees to the left around the scene (a three-quarter view from the left)",
    "right": "rotated about 45 degrees to the right around the scene (a three-quarter view from the right)",
    "left-profile": "moved 90 degrees to the left, seeing the scene from its left side (profile view)",
    "right-profile": "moved 90 degrees to the right, seeing the scene from its right side (profile view)",
    "front": "placed directly in front of the people, at eye level, so everyone faces the camera",
    "above": "raised high above, looking down at the scene at about 45 degrees",
    "below": "lowered near the ground, looking up at the scene (low angle)",
    "behind": "moved to the opposite side, looking at the scene from behind",
}

# Camera moves for orbit_video().
ORBITS: Dict[str, str] = {
    "orbit-left": "The camera slowly orbits around the scene to the left in a smooth arc, revealing it from new angles.",
    "orbit-right": "The camera slowly orbits around the scene to the right in a smooth arc, revealing it from new angles.",
    "push-in": "The camera slowly pushes in toward the people while drifting slightly sideways, revealing depth.",
    "crane-up": "The camera smoothly cranes up and over the scene, looking down at it from above.",
}

# Image-to-3D Spaces, best first. Each adapter returns the textured GLB path.
MESH_SPACES = ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2"]


def _slug(text: str, limit: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "photo"


def _target(source: Path, tag: str, suffix: str) -> Path:
    """Results are named after their source so they sort and read together."""
    stem = re.sub(r"^\d+-", "", source.stem)
    name = f"{int(time.time())}-{tag}-{stem}"[:120]
    return config.OUTPUT_DIR / (name + suffix)


def source_path(filename: str) -> Path:
    """Resolve a filename inside OUTPUT_DIR, refusing anything outside it."""
    path = (config.OUTPUT_DIR / filename).resolve()
    if config.OUTPUT_DIR.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"{filename} not found")
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("the source must be an image")
    return path


def _open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def save_upload(data: bytes, filename: str) -> Path:
    """Normalise an uploaded photo (EXIF rotation, RGB, <= MAX_SIDE) to a PNG."""
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("that file is not an image this server can read") from exc
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"{int(time.time())}-{UPLOAD_PREFIX}-{_slug(Path(filename).stem)}.png"
    image.save(path)
    return path


# --------------------------------------------------------------------------- #
# New viewpoint (Nano Banana)
# --------------------------------------------------------------------------- #
def view_prompt(angle: str, focus: str = "") -> str:
    move = ANGLES.get(angle, ANGLES["left"])
    prompt = (
        "This is a photograph. Re-render the exact same scene as if the camera had been "
        f"{move}. Keep every person and object from the original: the same people with the "
        "same faces, hair, skin tone, age, body shape, clothing and accessories, in the same "
        "positions relative to each other, with the same setting, lighting and time of day. "
        "Show surfaces that were hidden in the original in a way that is consistent with what "
        "is visible. Do not add or remove people or objects. Photorealistic, natural lens, "
        "no text, no borders."
    )
    if focus.strip():
        prompt += (
            f" Pay special attention to: {focus.strip()}. Make sure this person's face is clearly "
            "visible, sharp and well lit from this viewpoint, keeping their identity consistent "
            "with the original photo."
        )
    return prompt


def new_view(filename: str, angle: str, focus: str = "", model: Optional[str] = None) -> Path:
    source = source_path(filename)
    image = _open(source)
    model_id = model or config.VERTEX_IMAGE_MODELS[0]
    result = nano_banana(
        model_id, view_prompt(angle, focus), image.width, image.height, image=image
    )
    path = _target(source, f"view-{angle}", ".png")
    result.save(path)
    return path


# --------------------------------------------------------------------------- #
# Orbit clip (Veo image-to-video)
# --------------------------------------------------------------------------- #
def orbit_video(filename: str, move: str, focus: str = "", model: Optional[str] = None) -> Path:
    from google.genai import types

    from video_tool import veo

    source = source_path(filename)
    image = _open(source)
    # Veo is landscape or portrait only; it fits the photo into that frame.
    aspect = "9:16" if image.height > image.width else "16:9"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    prompt = (
        ORBITS.get(move, ORBITS["orbit-left"])
        + " Everything in the photo stays exactly as it is - the same people, faces, clothes "
        "and setting. The people stay still or move only slightly and naturally. "
        "Photorealistic, stable, no cuts, no text."
    )
    if focus.strip():
        prompt += f" The camera ends on {focus.strip()}, whose face becomes clearly visible."

    data = veo(
        model or config.VERTEX_VIDEO_MODELS[0],
        prompt,
        {
            "aspect_ratio": aspect,
            "duration_seconds": 8,
            "number_of_videos": 1,
            "person_generation": "allow_adult",
            "negative_prompt": "morphing faces, extra people, distorted bodies, flicker, text",
        },
        image=types.Image(image_bytes=buffer.getvalue(), mime_type="image/png"),
    )
    path = _target(source, f"orbit-{move}", ".mp4")
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# 3D mesh (Hunyuan3D on a free Space)
# --------------------------------------------------------------------------- #
def _hunyuan(client: Any, image: str, space: str) -> Any:
    from gradio_client import handle_file

    kwargs: Dict[str, Any] = dict(
        image=handle_file(image),
        mv_image_front=None,
        mv_image_back=None,
        mv_image_left=None,
        mv_image_right=None,
        steps=20,
        guidance_scale=5.0,
        seed=1234,
        octree_resolution=256,
        check_box_rembg=True,  # cut the subject out of the background first
        num_chunks=8000,
        randomize_seed=True,
    )
    if space == "tencent/Hunyuan3D-2":
        kwargs["caption"] = None
    # Returns (white mesh, textured mesh, html, stats, seed); we want the textured GLB.
    result = client.predict(**kwargs, api_name="/generation_all")
    return result[1]


_mesh_clients: Dict[str, Any] = {}


def to_mesh(filename: str) -> Path:
    from gradio_client import Client

    source = source_path(filename)
    errors: List[str] = []
    for space in MESH_SPACES:
        try:
            if space not in _mesh_clients:
                _mesh_clients[space] = Client(space, token=config.HF_TOKEN or None, verbose=False)
            glb = Path(result_path(_hunyuan(_mesh_clients[space], str(source), space)))
            path = _target(source, "3d", ".glb")
            shutil.copyfile(glb, path)
            return path
        except Exception as exc:  # try the next Space
            _mesh_clients.pop(space, None)
            errors.append(f"{space}: {type(exc).__name__}: {str(exc)[:200]}")

    if any("quota" in e.lower() or "runs limit" in e.lower() for e in errors):
        raise RuntimeError(config.quota_help("3D models"))
    raise RuntimeError(
        "The free 3D Spaces failed (they are sometimes queued or asleep). Details - "
        + " | ".join(errors)
    )


