"""FastAPI backend for the React web UI in frontend/.

    cd frontend && npm install && npm run build   # once, builds frontend/dist
    python server.py                              ->  http://localhost:8000
    uvicorn server:app --reload

For UI development run `npm run dev` in frontend/ as well: Vite serves the app on
:5173 with hot reload and proxies /api and /media to this server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import memory
import scene3d
from chain import MODES, enhance, memory_line, respond

# The built React app (frontend/, `npm run build`).
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

app = FastAPI(
    title="ImageBot", description="LangChain + Hugging Face image & video chatbot"
)

# Open-weights text-to-image models, newest and most prompt-accurate first.
# Those in config.SPACE_MAP also run on a free public Space.
# Google's paid Vertex models come first when a service-account key is set.
AVAILABLE_IMAGE_MODELS = [
    *(config.VERTEX_IMAGE_MODELS if config.HAS_VERTEX else []),
    "Tongyi-MAI/Z-Image-Turbo",
    "krea/Krea-2-Turbo",
    "black-forest-labs/FLUX.2-klein-4B",
    "Qwen/Qwen-Image-2512",
    "tencent/HunyuanImage-3.0",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
]
# Open-weights text-to-video models. Those in config.VIDEO_SPACE_MAP also run
# on a free public Space; the rest need a token that can call the Inference API.
AVAILABLE_VIDEO_MODELS = [
    *(config.VERTEX_VIDEO_MODELS if config.HAS_VERTEX else []),
    "MiniMaxAI/MiniMax-H3",
    "Wan-AI/Wan2.2-T2V-A14B",
    "tencent/HunyuanVideo-1.5",
    "Lightricks/LTX-Video-0.9.7-distilled",
]

# GIFs (demo-mode clips) are animated but play in an <img>, so they count as images.
VIDEO_EXTENSIONS = {".mp4", ".webm"}
MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".gif": "image/gif",
    ".glb": "model/gltf-binary",
}
# 3D models are served but not listed in the gallery (they have no thumbnail).
GALLERY_TYPES = {s for s in MEDIA_TYPES if s != ".glb"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return "model"
    return "video" if suffix in VIDEO_EXTENSIONS else "image"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    # Which conversation this turn belongs to. The server keeps the memory; the
    # browser only has to remember the id.
    session_id: str = "default"
    # Only used to re-seed a session the server no longer has (e.g. after a
    # restart) - see memory.prime_session.
    history: List[Turn] = Field(default_factory=list)
    # "image" or "video" - the switch in the UI.
    mode: str = "image"
    model: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    # Video only: clip length in seconds.
    duration: Optional[float] = None
    enhance: bool = True
    # Skip intent routing and draw every message, whatever it says.
    force_image: bool = True


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    models = list(dict.fromkeys([config.IMAGE_MODEL, *AVAILABLE_IMAGE_MODELS]))
    video_models = list(dict.fromkeys([config.VIDEO_MODEL, *AVAILABLE_VIDEO_MODELS]))
    return {
        "image_model": config.IMAGE_MODEL,
        "video_model": config.VIDEO_MODEL,
        "text_model": config.TEXT_MODEL,
        "backend": config.BACKEND,
        "has_token": bool(config.HF_TOKEN),
        "models": models,
        "video_models": video_models,
        # Models that can run with no token at all, via a public Space.
        "token_free_models": [m for m in models if m in config.SPACE_MAP],
        "token_free_video_models": [m for m in video_models if m in config.VIDEO_SPACE_MAP],
        # Google models on Vertex AI: they need the service-account key, not a HF token.
        "google_models": [
            m for m in models + video_models
            if config.HAS_VERTEX and m in config.VERTEX_IMAGE_MODELS + config.VERTEX_VIDEO_MODELS
        ],
        # How many messages LangChain keeps per session.
        "memory_window": memory.MAX_MESSAGES,
        "defaults": {
            model: {
                "steps": config.default_steps(model),
                "guidance_scale": config.default_guidance(model),
            }
            for model in models
        },
        "video_defaults": {
            model: {
                "steps": config.default_video_steps(model),
                "guidance_scale": config.default_video_guidance(model),
                "fps": config.video_fps(model),
            }
            for model in video_models
        },
        "video": {
            "duration": config.VIDEO_DURATION,
            "min_duration": config.VIDEO_MIN_DURATION,
            "max_duration": config.VIDEO_MAX_DURATION,
            "width": config.VIDEO_SIZE[0],
            "height": config.VIDEO_SIZE[1],
        },
        # Models the Inference API has refused for this token so far (403); their
        # results come from the free Space instead.
        "api_blocked": sorted(
            MODES["image"]["tool"]._api_blocked | MODES["video"]["tool"]._api_blocked
        ),
    }


@app.post("/api/chat")
def post_chat(req: ChatRequest) -> Dict[str, Any]:
    """One conversation turn: route -> (enhance -> generate) or reply with text."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(MODES)}")

    mode = req.mode
    model_id = req.model or MODES[mode]["model"]
    options: Dict[str, Any] = {
        "model": model_id,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "guidance_scale": req.guidance_scale,
        "seed": req.seed,
    }
    if mode == "video":
        options["duration"] = req.duration or config.VIDEO_DURATION
    # A restarted server has no memory of this session; the browser's copy of the
    # transcript refills it so follow-ups keep working across restarts.
    memory.prime_session(req.session_id, [turn.model_dump() for turn in req.history])

    try:
        if req.enhance:
            result = respond(
                req.message,
                session_id=req.session_id,
                force_image=req.force_image,
                mode=mode,
                **options,
            )
        else:
            # Bypass the LLM: the raw message goes straight to the diffusion model.
            # Still recorded, so the next enhanced turn can refer back to it.
            path = MODES[mode]["tool"].invoke({"prompt": req.message, **options})
            result = {"type": mode, "prompt": req.message, "path": path}
            memory.get_session_history(req.session_id).add_messages(
                memory.to_messages(
                    [
                        {"role": "user", "content": req.message},
                        {"role": "assistant", "content": memory_line(mode, req.message)},
                    ]
                )
            )
    except Exception as exc:  # surfaced in the UI as an error bubble
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    remembered = len(memory.session_messages(req.session_id))
    if result["type"] in MODES:
        path = Path(result["path"])
        tool = MODES[mode]["tool"]
        return {
            # "image" or "video", by what was actually written (demo mode writes GIFs).
            "type": media_kind(path),
            "prompt": result["prompt"],
            "url": f"/media/{path.name}",
            "filename": path.name,
            "model": model_id,
            "mode": mode,
            "duration": options.get("duration"),
            "remembered": remembered,
            # True when the token was refused by the Inference API and the free
            # Space produced this result instead.
            "api_blocked": model_id in tool._api_blocked,
        }
    return {"type": "text", "text": result["text"], "remembered": remembered}


@app.post("/api/enhance")
def post_enhance(req: ChatRequest) -> Dict[str, str]:
    """Preview what the LangChain prompt-enhancer would send to the diffusion model.

    Reads the session's memory but does not add to it - previewing a prompt is
    not a turn of the conversation.
    """
    payload = {
        "input": req.message,
        "history": memory.session_messages(req.session_id),
    }
    return {"prompt": enhance(payload, req.mode if req.mode in MODES else "image")}


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
class SessionRequest(BaseModel):
    session_id: str = "default"


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    """What the bot currently remembers about a conversation."""
    return {
        "session_id": session_id,
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in memory.session_messages(session_id)
        ],
        "window": memory.MAX_MESSAGES,
        "totals": memory.stats(),
    }


@app.post("/api/session/reset")
def reset_session(req: SessionRequest) -> Dict[str, Any]:
    """Forget a conversation."""
    return {"session_id": req.session_id, "cleared": memory.clear_session(req.session_id)}


@app.get("/api/gallery")
def get_gallery(limit: int = 24, kind: Optional[str] = None) -> Dict[str, Any]:
    """Recent generations, images and videos together, newest first.

    `kind` = "image" or "video" narrows the list.
    """
    files = [
        f
        for f in config.OUTPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in GALLERY_TYPES
    ]
    if kind in ("image", "video"):
        files = [f for f in files if media_kind(f) == kind]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    items = [
        {
            "type": media_kind(f),
            "url": f"/media/{f.name}",
            "filename": f.name,
            "created": f.stat().st_mtime,
        }
        for f in files
    ]
    return {"items": items}


# --------------------------------------------------------------------------- #
# 3D Studio: upload a photo, then new angles / orbit clip / 3D model
# --------------------------------------------------------------------------- #
def _item(path: Path) -> Dict[str, Any]:
    return {
        "type": media_kind(path),
        "url": f"/media/{path.name}",
        "filename": path.name,
        "created": path.stat().st_mtime,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Save a photo from the user's computer as a 3D Studio source."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="photos must be under 20 MB")
    try:
        path = scene3d.save_upload(data, file.filename or "photo")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return {**_item(path), "width": width, "height": height}


class Scene3DRequest(BaseModel):
    # A file in the output folder: an upload or any generated image.
    filename: str
    # "view": one of scene3d.ANGLES; "orbit": one of scene3d.ORBITS.
    angle: str = "left"
    move: str = "orbit-left"
    # Optional: which person to show clearly, e.g. "the man on the right".
    focus: str = ""


@app.get("/api/3d/options")
def scene3d_options() -> Dict[str, Any]:
    return {
        "angles": list(scene3d.ANGLES),
        "moves": list(scene3d.ORBITS),
        "google": config.HAS_VERTEX,
    }


@app.post("/api/3d/{tool}")
def scene3d_run(tool: str, req: Scene3DRequest) -> Dict[str, Any]:
    """Run one 3D Studio tool on a source photo. Slow: up to a few minutes."""
    if tool in ("view", "orbit") and not config.HAS_VERTEX:
        raise HTTPException(
            status_code=400,
            detail="New angles and orbit clips use Google's models; add the service-account key (key.json).",
        )
    try:
        if tool == "view":
            path = scene3d.new_view(req.filename, req.angle, req.focus)
        elif tool == "orbit":
            path = scene3d.orbit_video(req.filename, req.move, req.focus)
        elif tool == "mesh":
            path = scene3d.to_mesh(req.filename)
        else:
            raise HTTPException(status_code=404, detail=f"unknown tool {tool}")
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {**_item(path), "tool": tool}


def _serve(filename: str) -> FileResponse:
    path = (config.OUTPUT_DIR / filename).resolve()
    # Keep path traversal (../) out of the served directory.
    if config.OUTPUT_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type=media_type)


@app.get("/media/{filename}")
def get_media(filename: str) -> FileResponse:
    """Serve a generated image or video clip."""
    return _serve(filename)


@app.get("/images/{filename}")
def get_image(filename: str) -> FileResponse:
    """Older URL kept for transcripts browsers saved before video support."""
    return _serve(filename)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/", response_class=PlainTextResponse)
    def frontend_missing() -> str:
        return (
            "The web UI has not been built yet.\n\n"
            "    cd frontend && npm install && npm run build\n\n"
            "then restart the server. (For development, `npm run dev` serves it on :5173.)"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
