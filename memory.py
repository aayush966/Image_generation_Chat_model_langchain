"""Session memory for the chatbot, built on LangChain's chat-history API.

This is the modern replacement for the deprecated `ConversationBufferMemory`:
one `BaseChatMessageHistory` per session id, which `RunnableWithMessageHistory`
reads before a chain runs and writes after it finishes (see `chain.py`).

The store lives in this process, so it is per-run, not per-user-forever:
restart the server and every session starts empty.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Dict, Iterable, List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# A rolling window: how many messages (not turns) each session keeps.
# Diffusion follow-ups only ever refer to the last image or two, and a short
# window keeps the prompt small enough for the free Inference API.
MAX_MESSAGES = 16
# Cap on simultaneously remembered sessions, so a long-lived server cannot grow
# without bound. The least recently used session is dropped first.
MAX_SESSIONS = 200


class WindowedChatMessageHistory(InMemoryChatMessageHistory):
    """In-memory history that forgets everything before the last N messages."""

    max_messages: int = MAX_MESSAGES

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        super().add_messages(messages)
        overflow = len(self.messages) - self.max_messages
        if overflow > 0:
            del self.messages[:overflow]


# session id -> history, ordered least-recently-used first.
_sessions: "OrderedDict[str, WindowedChatMessageHistory]" = OrderedDict()
# FastAPI runs sync endpoints in a thread pool, so the store needs a lock.
_lock = Lock()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """History for one session, created on first use.

    This is the factory handed to `RunnableWithMessageHistory`; LangChain calls
    it with the `session_id` from each invocation's config.
    """
    with _lock:
        history = _sessions.get(session_id)
        if history is None:
            history = WindowedChatMessageHistory()
            _sessions[session_id] = history
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return history


def session_messages(session_id: str) -> List[BaseMessage]:
    """Everything currently remembered for a session (read-only use)."""
    return list(get_session_history(session_id).messages)


def clear_session(session_id: str) -> int:
    """Forget a session. Returns how many messages were dropped."""
    with _lock:
        history = _sessions.pop(session_id, None)
    return len(history.messages) if history else 0


def to_messages(turns: Iterable[Dict[str, str]]) -> List[BaseMessage]:
    """Convert the UI's `{role, content}` dicts into LangChain messages."""
    return [
        HumanMessage(content=turn["content"])
        if turn.get("role") == "user"
        else AIMessage(content=turn["content"])
        for turn in turns
        if turn.get("content")
    ]


def prime_session(session_id: str, turns: Iterable[Dict[str, str]]) -> int:
    """Seed an *empty* session from a transcript the client still holds.

    The browser keeps its own copy of the conversation in localStorage, so after
    a server restart the user can still see the chat. Without this the bot would
    have amnesia the user cannot see; with it, the first message after a restart
    quietly restores the window. Sessions that already have memory are left
    alone - the server's copy is the source of truth.
    """
    history = get_session_history(session_id)
    if history.messages:
        return 0
    messages = to_messages(turns)[-MAX_MESSAGES:]
    if messages:
        history.add_messages(messages)
    return len(messages)


def stats() -> Dict[str, int]:
    """Bookkeeping for the /api/session endpoints."""
    with _lock:
        return {
            "sessions": len(_sessions),
            "messages": sum(len(h.messages) for h in _sessions.values()),
        }
