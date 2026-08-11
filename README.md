# 🎨 ImageBot — LangChain + Hugging Face image-generation chatbot

A chatbot where you type a description and get an image back, built with **LangChain**
orchestration on top of **open-source Hugging Face models**.

## How it works

```
user message
   │
   ▼
LangChain router chain ──► "chat"  ──► conversation chain ──► text reply
   │                                        (Qwen2.5-7B-Instruct)
   └────────────────────► "image" ──► prompt-enhancer chain
                                            │  turns "a fox" into a full
                                            │  diffusion prompt
                                            ▼
                                    HuggingFaceImageTool (LangChain BaseTool)
                                            │
                                            ▼
                                    FLUX.1-schnell → PNG in ./generated
```

Everything is LCEL: `RunnablePassthrough.assign(intent=…) | RunnableLambda(dispatch)`,
with `ChatPromptTemplate` + `MessagesPlaceholder` giving each chain the conversation
history, so follow-ups like *"same but at night"* work.

| File | Role |
|---|---|
| `server.py` | FastAPI backend + JSON API for the web UI |
| `static/index.html` · `static/css/style.css` · `static/js/app.js` | The HTML/CSS/JS chat interface (no frameworks, no CDN) |
| `chain.py` | LangChain router / prompt-enhancer / chat chains + the pipeline |
| `memory.py` | LangChain session memory (`BaseChatMessageHistory` per session id) |
| `image_tool.py` | `HuggingFaceImageTool(BaseTool)` — the text-to-image tool |
| `app.py` | Alternative Streamlit UI (same chains) |
| `config.py` | env-driven configuration |

## Memory

The bot remembers the conversation with LangChain's own chat-history API rather than
by trusting whatever transcript the client sends:

```
respond("same but at night", session_id="abc")
   │
   ▼
RunnableWithMessageHistory
   │  ├─ before: loads session "abc" into the MessagesPlaceholder("history") slot
   │  └─ after:  appends this turn's human + AI message back into it
   ▼
router / enhancer / chat chains  ← all three see the same history
```

- `memory.py` keeps one `WindowedChatMessageHistory` (an `InMemoryChatMessageHistory`
  subclass) per session id, trimmed to the last **16 messages** so prompts stay small.
- Sessions are held in an LRU capped at 200, so a long-running server cannot grow
  without bound.
- For an image turn the assistant's remembered message is `Generated an image of: <prompt>`
  — storing the *prompt* rather than the file path is what lets *"same but at night"*
  resolve against the right picture.
- A turn that raises is never written to memory, so a failed generation leaves no trace.
- Memory is **per process**: restart the server and it is gone. The browser also keeps
  the transcript in `localStorage` and sends it with the next message, which re-seeds an
  empty session — so a restart is invisible to the user.

Each browser gets a `session_id` (a UUID in `localStorage`), so two tabs are two
independent conversations. The sidebar shows how full the memory window is.

| Endpoint | Purpose |
|---|---|
| `GET /api/session/{id}` | what the bot currently remembers |
| `POST /api/session/reset` | forget a conversation (wired to **Clear chat**) |

In the terminal chat (`python chain.py`) type `forget` to clear the session.

### Web UI features

Dark/light theme, model picker, size / steps / guidance / seed controls, prompt-enhance
toggle, streaming-style typing indicator, image cards with **Download / Copy prompt /
Variation**, click-to-zoom lightbox, a gallery of past generations, a live memory
counter, a transcript restored on reload, and a responsive layout with a collapsible
sidebar on mobile.

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | one turn — takes `session_id`, returns `{type:"image", prompt, url, filename, model, remembered}` or `{type:"text", text, remembered}` |
| `POST /api/enhance` | preview the LangChain-enhanced prompt without generating (reads memory, does not write to it) |
| `GET /api/config` | models, backend, token status, per-model defaults, memory window |
| `GET /api/gallery` | recent generations |
| `GET /api/session/{id}` | the remembered messages for a session |
| `POST /api/session/reset` | clear a session's memory |
| `GET /images/{file}` | serve a generated PNG |

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env        # Windows  (cp on macOS/Linux)
```

Put a free Hugging Face **read** token in `.env`
(https://huggingface.co/settings/tokens):

```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## Run

```bash
python server.py         # HTML/CSS/JS web UI  →  http://127.0.0.1:8000
streamlit run app.py     # alternative Streamlit UI → http://localhost:8501
python chain.py          # same bot in the terminal
```

### Backends

| `BACKEND` | Real images? | Token? | Notes |
|---|---|---|---|
| `auto` *(default)* | yes | optional | Inference API if a token is set, else the public Space |
| `api` | yes | **required** | HF Inference API; falls back to the Space if it errors |
| `space` | yes | optional | Runs FLUX.1-schnell on a public HF Space. **Free anonymous quota is only a few images per hour** — a token raises it a lot |
| `local` | yes | no | `diffusers` on your own GPU; downloads ~10–20 GB first |
| `demo` | no | no | Labelled placeholder images, for testing the UI offline |

So the app produces real images with no setup at all — but only a handful before the
free Space quota runs out. Adding a token is the fix, and it is free.

```bash
$env:BACKEND="demo"; python server.py   # exercise the UI with zero network calls
```

## Models

All open-weights, served by the HF Inference API by default:

- `black-forest-labs/FLUX.1-schnell` (default — 4 steps, fast, Apache-2.0)
- `stabilityai/stable-diffusion-xl-base-1.0`
- `stabilityai/stable-diffusion-3.5-large-turbo`
- `black-forest-labs/FLUX.1-dev`
- text side: `Qwen/Qwen2.5-7B-Instruct` (swap via `TEXT_MODEL`)

Switch models in the sidebar or in `.env`.

## Running the diffusion model locally (no API calls)

```bash
pip install diffusers torch transformers accelerate
```

then set `BACKEND=local` in `.env`. First run downloads the weights (~10–20 GB for
FLUX); a GPU is strongly recommended — on CPU a single image takes several minutes.

## Notes

- Generated PNGs are saved to `./generated/` and are downloadable from the chat.
- **Enhance prompt** off in the sidebar sends your text verbatim to the diffusion model
  and skips the LLM entirely.
- If the text model is unreachable, the bot degrades gracefully: a keyword heuristic
  routes the turn and your raw prompt is used for generation.
- FLUX models ignore `negative_prompt` and use `guidance_scale=0`; SD models use both.
