"""LangChain pieces of the chatbot.

Flow for every user turn:

    user text ──> router chain ──┬── "image" ──> prompt-enhancer chain ──┬─ mode "image" ─> HuggingFaceImageTool
                                 │                                      └─ mode "video" ─> HuggingFaceVideoTool
                                 └── "chat"  ──> conversational chain

The mode (image or video) is a switch the caller flips; the router only decides
whether the user wants *something generated* or is just talking.

All three chains run on an open-source instruct model served by the Hugging Face
Inference API through `langchain_huggingface`.

Conversation memory is LangChain's own: the pipeline is wrapped in
`RunnableWithMessageHistory`, which loads the session's past messages into the
`MessagesPlaceholder("history")` slot before the chains run and appends this
turn afterwards. Callers pass a `session_id` instead of a transcript.
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import config
import memory
from image_tool import image_tool
from video_tool import video_tool

# The two generation modes. Everything mode-specific is looked up here.
MODES = {
    "image": {
        "tool": image_tool,
        "noun": "image",
        "verb": "Generated an image of",
        "model": config.IMAGE_MODEL,
    },
    "video": {
        "tool": video_tool,
        "noun": "video clip",
        "verb": "Generated a video of",
        "model": config.VIDEO_MODEL,
    },
}

# LangChain nudges you toward LangGraph checkpointers, which would mean an extra
# dependency and a graph runtime for what is a two-branch chain. RunnableWithMessageHistory
# is still the supported chat-history API in langchain-core, so we keep it - and
# silence the notice, which would otherwise print on every single turn.
warnings.filterwarnings(
    "ignore",
    message=".*RunnableWithMessageHistory.*",
    category=Warning,
)

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_llm() -> ChatHuggingFace:
    """Open-source instruct model, served by the HF Inference API."""
    if not config.HF_TOKEN:
        raise RuntimeError(
            "No Hugging Face token found. Add HUGGINGFACEHUB_API_TOKEN to your .env file."
        )
    endpoint = HuggingFaceEndpoint(
        repo_id=config.TEXT_MODEL,
        task="text-generation",
        huggingfacehub_api_token=config.HF_TOKEN,
        max_new_tokens=512,
        temperature=0.6,
        repetition_penalty=1.03,
    )
    return ChatHuggingFace(llm=endpoint)


# --------------------------------------------------------------------------- #
# 1. Router: is this turn an image request or ordinary chat?
# --------------------------------------------------------------------------- #

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify the user's latest message in an image- and video-generation chatbot.\n"
            'Reply with JSON only: {{"intent": "image"}} or {{"intent": "chat"}}.\n'
            '- "image": the user wants a picture or video clip created, or wants the previous '
            "one changed/redone (e.g. \"make it night time\", \"same but a cat\"). A bare "
            "description of a subject or scene with no question in it is ALWAYS a generation "
            'request - "watercolour lighthouse during a storm" is "image", not "chat".\n'
            '- "chat": only greetings, thanks, and questions aimed at the assistant '
            '("who are you?", "which model do you use?", "how does this work?").\n'
            'When in doubt, answer "image".\n'
            "No explanation, no markdown fences.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

IMAGE_HINT = re.compile(
    r"\b(draw|paint|generate|create|make|render|design|illustrate|sketch|show me|picture|"
    r"image|photo|art|logo|poster|wallpaper|portrait|scene|video|clip|animate|animation|"
    r"film|footage|timelapse|time-lapse)\b",
    re.I,
)

# Small talk and questions aimed at the bot itself - everything else in an
# image-generation chatbot is assumed to be a description to draw.
CHAT_HINT = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ty|ok|okay|cool|nice|great|bye|goodbye|"
    r"who are you|what are you|what can you (do|make)|how (are|do) you|help|test)\b"
    # allow a short pleasantry after the greeting ("hi there", "thanks a lot")
    r"[\s\w']{0,14}[!.?]*$",
    re.I,
)


def _parse_intent(raw: str) -> str:
    match = re.search(r'"intent"\s*:\s*"(image|chat)"', raw, re.I)
    if match:
        return match.group(1).lower()
    return "image" if "image" in raw.lower() else "chat"


def heuristic_intent(text: str) -> str:
    """Keyword routing used when the LLM is unreachable.

    Biased toward "image": this is an image generator, so a plain description
    such as "watercolour lighthouse during a storm" must be drawn, not chatted at.
    """
    stripped = text.strip()
    if IMAGE_HINT.search(stripped):
        return "image"
    if CHAT_HINT.match(stripped):
        return "chat"
    # Prompts are descriptions, not questions - anything ending in "?" is talk.
    if stripped.endswith("?"):
        return "chat"
    # Two or more words: treat it as a description to draw.
    return "image" if len(stripped.split()) >= 2 else "chat"


def build_router():
    return ROUTER_PROMPT | get_llm() | StrOutputParser() | RunnableLambda(_parse_intent)


def route(payload: Dict[str, Any]) -> str:
    """Classify a turn, falling back to a keyword heuristic if the LLM is unreachable."""
    try:
        return build_router().invoke(payload)
    except Exception:
        return heuristic_intent(payload["input"])


# --------------------------------------------------------------------------- #
# 2. Prompt enhancer: short user request -> rich diffusion prompt
# --------------------------------------------------------------------------- #

ENHANCER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write prompts for a text-to-image diffusion model.\n"
            "Expand the user's request into ONE vivid English prompt of 30-60 words "
            "covering: subject, action, setting, lighting, colour palette, camera or art "
            "style, and level of detail.\n"
            "If the user is refining a previous image, carry over everything from that "
            "image except what they asked to change.\n"
            "Output the prompt text only - no quotes, no labels, no explanation.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


VIDEO_ENHANCER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write prompts for a text-to-video diffusion model.\n"
            "Expand the user's request into ONE vivid English prompt of 40-70 words that "
            "describes a short clip: the subject and what it is DOING, how things move, "
            "the camera (static, slow pan, dolly in, handheld...), setting, lighting, "
            "colour palette and visual style. Keep it to a single continuous shot.\n"
            "If the user is refining a previous clip, carry over everything from that clip "
            "except what they asked to change.\n"
            "Output the prompt text only - no quotes, no labels, no explanation.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

ENHANCER_PROMPTS = {"image": ENHANCER_PROMPT, "video": VIDEO_ENHANCER_PROMPT}


def _clean_prompt(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(prompt|final prompt|image prompt)\s*[:\-]\s*", "", text, flags=re.I)
    return " ".join(text.split())


def build_enhancer(mode: str = "image"):
    prompt = ENHANCER_PROMPTS.get(mode, ENHANCER_PROMPT)
    return prompt | get_llm() | StrOutputParser() | RunnableLambda(_clean_prompt)


def enhance(payload: Dict[str, Any], mode: str = "image") -> str:
    """Enhanced prompt, or the raw user text if the LLM is unavailable."""
    try:
        enhanced = build_enhancer(mode).invoke(payload)
        return enhanced if len(enhanced) > 10 else payload["input"]
    except Exception:
        return payload["input"]


# --------------------------------------------------------------------------- #
# 3. Plain conversation
# --------------------------------------------------------------------------- #

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are ImageBot, a friendly assistant that generates images and short "
            f"video clips from descriptions, using the open-source models {config.IMAGE_MODEL} "
            f"(images) and {config.VIDEO_MODEL} (video). The user picks the mode with the "
            "Image / Video switch in the app. Keep replies to two or three sentences and "
            "nudge the user toward describing a picture or clip you could create for them.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


def build_chat_chain():
    return CHAT_PROMPT | get_llm() | StrOutputParser()


def chat(payload: Dict[str, Any]) -> str:
    try:
        return build_chat_chain().invoke(payload).strip()
    except Exception as exc:
        if "sufficient permissions" in str(exc) or "403" in str(exc):
            return (
                "My chat model is refusing the request: your Hugging Face token does "
                "not have the \"Make calls to Inference Providers\" permission. Enable "
                "it at https://huggingface.co/settings/tokens (or create a classic "
                "Read token) and restart the server. Image generation works either way."
            )
        if not config.HF_TOKEN:
            return (
                "I'm running without a Hugging Face token, so the chat model is offline - "
                "add HUGGINGFACEHUB_API_TOKEN to your .env file to switch it on. "
                "Image generation still works: just describe a picture, "
                "e.g. \"a watercolour lighthouse during a storm\"."
            )
        return (
            "I couldn't reach the chat model just now, but image generation still works - "
            "describe the picture you'd like."
        )


# --------------------------------------------------------------------------- #
# The full pipeline
# --------------------------------------------------------------------------- #


def build_pipeline(force_image: bool = False, mode: str = "image", **generation_options):
    """LCEL pipeline: route -> (enhance -> generate) | chat, with memory.

    With force_image=True the router is skipped entirely and every message is
    drawn - the "whatever I type, make a picture of it" mode.

    `mode` picks the generator: "image" (HuggingFaceImageTool) or "video"
    (HuggingFaceVideoTool). The remaining keyword arguments are passed straight
    to that tool.

    The result is wrapped in `RunnableWithMessageHistory`, so the caller only
    supplies `{"input": ...}` plus a session id; the `history` slot every prompt
    template declares is filled in by LangChain from that session's memory.
    """

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    spec = MODES[mode]

    def _dispatch(data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"input": data["input"], "history": data["history"]}

        if data["intent"] != "image":
            reply = chat(payload)
            return {"type": "text", "text": reply, "memory": reply}

        prompt = enhance(payload, mode)
        path = spec["tool"].invoke({"prompt": prompt, **generation_options})
        return {
            "type": mode,
            "prompt": prompt,
            "path": path,
            # What the assistant "said" this turn, as far as memory is concerned.
            # Storing the prompt rather than the file path is what makes refinements
            # like "same but at night" resolve against the right picture or clip.
            "memory": memory_line(mode, prompt),
        }

    router = (
        RunnableLambda(lambda _: "image") if force_image else RunnableLambda(route)
    )
    pipeline = RunnablePassthrough.assign(intent=router) | RunnableLambda(_dispatch)

    return RunnableWithMessageHistory(
        pipeline,
        memory.get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="memory",
    )


def memory_line(mode: str, prompt: str) -> str:
    """The assistant message remembered for a generation turn."""
    return f"{MODES[mode]['verb']}: {prompt}"


def respond(
    user_input: str,
    session_id: str = "default",
    force_image: bool = False,
    mode: str = "image",
    **generation_options,
) -> Dict[str, Any]:
    """One turn of the chatbot, remembered under `session_id`.

    `mode` is "image" or "video". Nothing is written to memory if the turn
    raises - a failed generation does not pollute the history.
    """
    return build_pipeline(force_image=force_image, mode=mode, **generation_options).invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":  # quick terminal chat: python chain.py
    SESSION = "terminal"
    mode = "image"
    print(
        f"ImageBot - images: {config.IMAGE_MODEL} | video: {config.VIDEO_MODEL}\n"
        "Type 'video' or 'image' to switch mode, 'forget' to clear memory, 'exit' to quit.\n"
    )
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "forget":
            print(f"Bot: forgot {memory.clear_session(SESSION)} messages.\n")
            continue
        if user.lower() in MODES:
            mode = user.lower()
            print(f"Bot: switched to {mode} mode ({MODES[mode]['model']}).\n")
            continue
        try:
            result = respond(user, session_id=SESSION, mode=mode)
        except Exception as exc:
            print(f"Bot: error - {exc}\n")
            continue
        if result["type"] in MODES:
            reply = f"Saved {result['path']}\n     prompt: {result['prompt']}"
        else:
            reply = result["text"]
        remembered = len(memory.session_messages(SESSION))
        print(f"Bot: {reply}\n     [memory: {remembered} messages]\n")
