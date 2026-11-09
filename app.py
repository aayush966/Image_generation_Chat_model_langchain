"""Streamlit chat UI for the LangChain + Hugging Face image & video generator.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

import config
import memory
from chain import MODES, memory_line, respond

st.set_page_config(page_title="ImageBot", page_icon="🎨", layout="centered")

# One LangChain memory session per browser tab. Streamlit reruns the whole
# script on every interaction, so the id has to live in session_state.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    # The Image / Video switch: which Hugging Face model the prompt goes to.
    mode = st.radio(
        "Generate", ["image", "video"], horizontal=True, format_func=str.title
    )
    is_video = mode == "video"

    if is_video:
        model = st.selectbox(
            "Video model (open source)",
            list(dict.fromkeys([
                config.VIDEO_MODEL,
                "MiniMaxAI/MiniMax-H3",
                "Wan-AI/Wan2.2-T2V-A14B",
                "tencent/HunyuanVideo-1.5",
                "Lightricks/LTX-Video-0.9.7-distilled",
            ])),
            index=0,
            help="MiniMax-H3 and LTX-Video run on free public Spaces; the rest need API access.",
        )
        size = st.select_slider(
            "Frame size", options=["544x544", "960x544", "544x960", "768x768", "1280x704", "704x1280"],
            value="960x544",
        )
        duration = st.slider("Clip length (seconds)", config.VIDEO_MIN_DURATION, config.VIDEO_MAX_DURATION, config.VIDEO_DURATION, 1.0)
        steps = st.slider("Steps", 1, 50, config.default_video_steps(model))
        guidance = st.slider("Guidance scale", 0.0, 15.0, config.default_video_guidance(model), 0.5)
    else:
        model = st.selectbox(
            "Image model (open source)",
            list(dict.fromkeys([
                config.IMAGE_MODEL,
                "Tongyi-MAI/Z-Image-Turbo",
                "krea/Krea-2-Turbo",
                "black-forest-labs/FLUX.2-klein-4B",
                "Qwen/Qwen-Image-2512",
                "tencent/HunyuanImage-3.0",
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-schnell",
            ])),
            index=0,
            help="Z-Image, Krea 2 and FLUX.2 klein run on free public Spaces; the rest need API access.",
        )
        size = st.select_slider(
            "Size", options=["512x512", "768x768", "1024x1024", "1024x576", "576x1024"],
            value="1024x1024",
        )
        duration = None
        steps = st.slider("Steps", 1, 50, config.default_steps(model))
        guidance = st.slider("Guidance scale", 0.0, 15.0, config.default_guidance(model), 0.5)
    width, height = (int(v) for v in size.split("x"))

    seed_text = st.text_input("Seed (blank = random)", "")
    enhance_prompt = st.toggle("Enhance prompt with the LLM", value=True)

    st.divider()
    st.caption(f"Backend: `{config.BACKEND}`  ·  Text model: `{config.TEXT_MODEL}`")
    st.caption("Token loaded ✅" if config.HF_TOKEN else "⚠️ No HF token in .env")
    remembered = len(memory.session_messages(st.session_state.session_id))
    st.caption(f"🧠 Memory: {remembered}/{memory.MAX_MESSAGES} messages")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        memory.clear_session(st.session_state.session_id)
        st.rerun()

# --------------------------------------------------------------------------- #
# Chat state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎬 ImageBot" if is_video else "🎨 ImageBot")
st.caption(
    "Describe a short clip and I'll render it with an open-source Hugging Face text-to-video model."
    if is_video
    else "Describe a picture and I'll generate it with an open-source Hugging Face model."
)


def show_media(path: str, key: str) -> None:
    """Render a generated file inline: MP4s as video, everything else as an image."""
    file = Path(path)
    if not file.exists():
        return
    is_clip = file.suffix.lower() in {".mp4", ".webm"}
    if is_clip:
        st.video(str(file))
    else:
        st.image(str(file), use_container_width=True)
    mime = {".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif"}.get(
        file.suffix.lower(), "image/png"
    )
    with open(file, "rb") as fh:
        st.download_button("⬇️ Download", fh.read(), file_name=file.name, mime=mime, key=key)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        path = message.get("media") or message.get("image")
        if path:
            show_media(path, key=f"dl-{path}")

# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
placeholder = (
    "e.g. a red fox trotting through a snowy forest, slow tracking shot"
    if is_video
    else "e.g. a red fox sleeping in a snowy forest at dawn"
)
if user_input := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
    options = {
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    }
    if is_video:
        options["duration"] = duration

    with st.chat_message("assistant"):
        with st.spinner("Rendering video (a minute or two)…" if is_video else "Thinking…"):
            try:
                if enhance_prompt:
                    # History comes from LangChain's memory, keyed by session id.
                    result = respond(
                        user_input, session_id=st.session_state.session_id, mode=mode, **options
                    )
                else:
                    # Skip the LLM entirely: send the raw text straight to the model.
                    path = MODES[mode]["tool"].invoke({"prompt": user_input, **options})
                    result = {"type": mode, "prompt": user_input, "path": path}
                    memory.get_session_history(st.session_state.session_id).add_messages(
                        memory.to_messages(
                            [
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": memory_line(mode, user_input)},
                            ]
                        )
                    )
            except Exception as exc:
                result = {"type": "text", "text": f"❌ {type(exc).__name__}: {exc}"}

        if result["type"] in MODES:
            reply = f"Here it is 👇\n\n**Prompt used:** _{result['prompt']}_"
            st.markdown(reply)
            show_media(result["path"], key=f"dl-new-{result['path']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "media": result["path"]}
            )
        else:
            st.markdown(result["text"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["text"]}
            )
