"""FastAPI backend for the HTML/CSS/JS chat UI.

    python server.py            ->  http://localhost:8000
    uvicorn server:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from chain import chat as chat_chain
from chain import enhance, respond, to_messages
from image_tool import image_tool

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ImageBot", description="LangChain + Hugging Face image chatbot")

AVAILABLE_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-large-turbo",
]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Turn] = Field(default_factory=list)
    model: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    enhance: bool = True
    # Skip intent routing and draw every message, whatever it says.
    force_image: bool = True


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    models = list(dict.fromkeys([config.IMAGE_MODEL, *AVAILABLE_IMAGE_MODELS]))
    return {
        "image_model": config.IMAGE_MODEL,
        "text_model": config.TEXT_MODEL,
        "backend": config.BACKEND,
        "has_token": bool(config.HF_TOKEN),
        "models": models,
        # Models that can run with no token at all, via a public Space.
        "token_free_models": [m for m in models if m in config.SPACE_MAP],
        "defaults": {
            model: {
                "steps": config.default_steps(model),
                "guidance_scale": config.default_guidance(model),
            }
            for model in models
        },
    }


@app.post("/api/chat")
def post_chat(req: ChatRequest) -> Dict[str, Any]:
    """One conversation turn: route -> (enhance -> generate) or reply with text."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    model_id = req.model or config.IMAGE_MODEL
    options = {
        "model": model_id,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "guidance_scale": req.guidance_scale,
        "seed": req.seed,
    }
    history = [turn.model_dump() for turn in req.history]

    try:
        if req.enhance:
            result = respond(req.message, history, force_image=req.force_image, **options)
        else:
            # Bypass the LLM: the raw message goes straight to the diffusion model.
            path = image_tool.invoke({"prompt": req.message, **options})
            result = {"type": "image", "prompt": req.message, "path": path}
    except Exception as exc:  # surfaced in the UI as an error bubble
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    if result["type"] == "image":
        name = Path(result["path"]).name
        return {
            "type": "image",
            "prompt": result["prompt"],
            "url": f"/images/{name}",
            "filename": name,
            "model": model_id,
        }
    return {"type": "text", "text": result["text"]}


@app.post("/api/enhance")
def post_enhance(req: ChatRequest) -> Dict[str, str]:
    """Preview what the LangChain prompt-enhancer would send to the diffusion model."""
    payload = {
        "input": req.message,
        "history": to_messages([turn.model_dump() for turn in req.history]),
    }
    return {"prompt": enhance(payload)}


@app.get("/api/gallery")
def get_gallery(limit: int = 24) -> Dict[str, Any]:
    files = sorted(
        config.OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:limit]
    return {
        "images": [
            {"url": f"/images/{f.name}", "filename": f.name, "created": f.stat().st_mtime}
            for f in files
        ]
    }


@app.get("/images/{filename}")
def get_image(filename: str) -> FileResponse:
    path = (config.OUTPUT_DIR / filename).resolve()
    # Keep path traversal (../) out of the served directory.
    if config.OUTPUT_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
