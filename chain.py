"""LangChain pieces of the chatbot.

Flow for every user turn:

    user text ──> router chain ──┬── "image" ──> prompt-enhancer chain ──> HuggingFaceImageTool
                                 └── "chat"  ──> conversational chain

All three chains run on an open-source instruct model served by the Hugging Face
Inference API through `langchain_huggingface`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import config
from image_tool import image_tool

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
            "You classify the user's latest message in an image-generation chatbot.\n"
            'Reply with JSON only: {{"intent": "image"}} or {{"intent": "chat"}}.\n'
            '- "image": the user wants a picture created, or wants the previous picture '
            "changed/redone (e.g. \"make it night time\", \"same but a cat\"). A bare "
            "description of a subject or scene with no question in it is ALWAYS an image "
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
    r"image|photo|art|logo|poster|wallpaper|portrait|scene)\b",
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


def _clean_prompt(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"^(prompt|final prompt|image prompt)\s*[:\-]\s*", "", text, flags=re.I)
    return " ".join(text.split())


def build_enhancer():
    return ENHANCER_PROMPT | get_llm() | StrOutputParser() | RunnableLambda(_clean_prompt)


def enhance(payload: Dict[str, Any]) -> str:
    """Enhanced prompt, or the raw user text if the LLM is unavailable."""
    try:
        enhanced = build_enhancer().invoke(payload)
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
            "You are ImageBot, a friendly assistant that generates images from "
            f"descriptions using the open-source model {config.IMAGE_MODEL}. "
            "Keep replies to two or three sentences and nudge the user toward "
            "describing a picture you could create for them.",
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


def to_messages(history: List[Dict[str, str]], limit: int = 8):
    """Convert the UI's history dicts into LangChain messages."""
    messages = []
    for turn in history[-limit:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def build_pipeline(force_image: bool = False, **image_options):
    """LCEL pipeline: route -> (enhance -> generate) | chat.

    With force_image=True the router is skipped entirely and every message is
    drawn - the "whatever I type, make a picture of it" mode.
    """

    def _dispatch(data: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"input": data["input"], "history": data["history"]}

        if data["intent"] != "image":
            return {"type": "text", "text": chat(payload)}

        prompt = enhance(payload)
        path = image_tool.invoke({"prompt": prompt, **image_options})
        return {"type": "image", "prompt": prompt, "path": path}

    router = (
        RunnableLambda(lambda _: "image") if force_image else RunnableLambda(route)
    )
    return RunnablePassthrough.assign(intent=router) | RunnableLambda(_dispatch)


def respond(
    user_input: str,
    history: List[Dict[str, str]] | None = None,
    force_image: bool = False,
    **image_options,
) -> Dict[str, Any]:
    """One turn of the chatbot."""
    return build_pipeline(force_image=force_image, **image_options).invoke(
        {"input": user_input, "history": to_messages(history or [])}
    )


if __name__ == "__main__":  # quick terminal chat: python chain.py
    turns: List[Dict[str, str]] = []
    print(f"ImageBot ({config.IMAGE_MODEL}) - type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        try:
            result = respond(user, turns)
        except Exception as exc:
            print(f"Bot: error - {exc}\n")
            continue
        if result["type"] == "image":
            reply = f"Saved {result['path']}\n     prompt: {result['prompt']}"
        else:
            reply = result["text"]
        print(f"Bot: {reply}\n")
        turns += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(result)[:400]},
        ]
