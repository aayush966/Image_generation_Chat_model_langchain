"""Streamlit chat UI for the LangChain + Hugging Face image generator.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

import config
import memory
from chain import respond
from image_tool import image_tool

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

    image_model = st.selectbox(
        "Image model (open source)",
        [
            config.IMAGE_MODEL,
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "black-forest-labs/FLUX.1-dev",
        ],
        index=0,
    )
    image_tool.model_id = image_model

    size = st.select_slider(
        "Size", options=["512x512", "768x768", "1024x1024", "1024x576", "576x1024"],
        value="1024x1024",
    )
    width, height = (int(v) for v in size.split("x"))

    steps = st.slider("Steps", 1, 50, config.default_steps(image_model))
    guidance = st.slider("Guidance scale", 0.0, 15.0, config.default_guidance(image_model), 0.5)
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

st.title("🎨 ImageBot")
st.caption("Describe a picture and I'll generate it with an open-source Hugging Face model.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        path = message.get("image")
        if path and Path(path).exists():
            st.image(path, use_container_width=True)
            with open(path, "rb") as fh:
                st.download_button(
                    "⬇️ Download",
                    fh.read(),
                    file_name=Path(path).name,
                    mime="image/png",
                    key=f"dl-{path}",
                )

# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
if user_input := st.chat_input("e.g. a red fox sleeping in a snowy forest at dawn"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    seed = int(seed_text) if seed_text.strip().lstrip("-").isdigit() else None
    options = {
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if enhance_prompt:
                    # History comes from LangChain's memory, keyed by session id.
                    result = respond(
                        user_input, session_id=st.session_state.session_id, **options
                    )
                else:
                    # Skip the LLM entirely: send the raw text straight to the model.
                    path = image_tool.invoke({"prompt": user_input, **options})
                    result = {"type": "image", "prompt": user_input, "path": path}
                    memory.get_session_history(st.session_state.session_id).add_messages(
                        memory.to_messages(
                            [
                                {"role": "user", "content": user_input},
                                {
                                    "role": "assistant",
                                    "content": f"Generated an image of: {user_input}",
                                },
                            ]
                        )
                    )
            except Exception as exc:
                result = {"type": "text", "text": f"❌ {type(exc).__name__}: {exc}"}

        if result["type"] == "image":
            reply = f"Here it is 👇\n\n**Prompt used:** _{result['prompt']}_"
            st.markdown(reply)
            st.image(result["path"], use_container_width=True)
            with open(result["path"], "rb") as fh:
                st.download_button(
                    "⬇️ Download",
                    fh.read(),
                    file_name=Path(result["path"]).name,
                    mime="image/png",
                    key=f"dl-new-{result['path']}",
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "image": result["path"]}
            )
        else:
            st.markdown(result["text"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["text"]}
            )
