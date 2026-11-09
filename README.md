# 🎨 ImageBot — LangChain + Hugging Face image & video chatbot

A chatbot where you type a description and get an **image** or a short **video clip**
back, built with **LangChain** orchestration on top of **open-source Hugging Face models**.
An **Image / Video** switch at the top of the stage picks which model the next prompt
goes to.

## How it works

```
user message  +  mode switch (image | video)
   │
   ▼
LangChain router chain ──► "chat"  ──► conversation chain ──► text reply
   │                                        (Qwen2.5-7B-Instruct)
   └────────────────────► "image" ──► prompt-enhancer chain
                                      (image prompt or video prompt, by mode)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                 HuggingFaceImageTool            HuggingFaceVideoTool
                 Z-Image-Turbo → PNG             MiniMax-H3 → MP4
                              └──────── ./generated ──────┘
```

Everything is LCEL: `RunnablePassthrough.assign(intent=…) | RunnableLambda(dispatch)`,
with `ChatPromptTemplate` + `MessagesPlaceholder` giving each chain the conversation
history, so follow-ups like *"same but at night"* work.

| File | Role |
|---|---|
| `server.py` | FastAPI backend + JSON API for the web UI |
| `frontend/` | The web UI: React 19 + Vite + TypeScript + Tailwind CSS (`src/store.tsx` holds state and API calls, `src/components/` the panes) |
| `chain.py` | LangChain router / prompt-enhancer / chat chains + the pipeline |
| `memory.py` | LangChain session memory (`BaseChatMessageHistory` per session id) |
| `image_tool.py` | `HuggingFaceImageTool(BaseTool)` — the text-to-image tool |
| `video_tool.py` | `HuggingFaceVideoTool(BaseTool)` — the text-to-video tool (same backends) |
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
- For a generation turn the assistant's remembered message is
  `Generated an image of: <prompt>` or `Generated a video of: <prompt>` — storing the
  *prompt* rather than the file path is what lets *"same but at night"* resolve against
  the right picture or clip, and lets you draw something and then animate it.
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

In the terminal chat (`python chain.py`) type `forget` to clear the session and
`video` / `image` to flip the mode.

### Web UI features

A top bar (Create · Gallery · 3D Studio, server status, New chat, theme), a settings
panel (Image / Video switch — also `Ctrl+M` — model, aspect ratio, resolution, clip
length, with steps / guidance / seed / always-generate under **Advanced**) and a
workspace that shows the latest picture or clip with **Vary / Copy / 3D / Download**,
a strip of this session's results, and the prompt box with the enhance toggle.
Click-to-zoom lightbox, a full gallery, dark/light theme, the last result restored on
reload, and a slide-in settings drawer on mobile.

### 3D Studio

Upload a photo (or send any generated image there with the **3D** button) and:

| Tool | Model | What you get |
|---|---|---|
| **Angles** | Nano Banana 2 (Vertex, paid) | the same scene re-rendered from left / right / side / front / behind / above / low; optionally name a person to turn their face to the camera |
| **Orbit** | Veo 3.1 Fast (Vertex, paid) | an 8 s image-to-video camera move around the scene |
| **3D** | Hunyuan3D-2.1 on a free Space | a textured `.glb` you can drag to rotate in the browser — best for one person or object, not a group scene |

A single photo does not contain what the camera never saw: a turned-away face is
**invented** by the model to match what is visible, not recovered.

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | one turn — takes `session_id`, `mode` (`image` or `video`) and, for video, `duration`; returns `{type:"image"|"video", prompt, url, filename, model, mode, duration, remembered}` or `{type:"text", text, remembered}` |
| `POST /api/enhance` | preview the LangChain-enhanced prompt without generating (reads memory, does not write to it); uses the video prompt template when `mode` is `video` |
| `GET /api/config` | image and video model lists, backend, token status, per-model defaults, video defaults, memory window |
| `GET /api/gallery` | recent generations, images and clips together (`?kind=video` or `?kind=image` to filter) |
| `GET /api/session/{id}` | the remembered messages for a session |
| `POST /api/session/reset` | clear a session's memory |
| `POST /api/upload` | multipart `file` → saves a photo as a 3D Studio source |
| `POST /api/3d/{view,orbit,mesh}` | `{filename, angle?, move?, focus?}` → a new angle (PNG), orbit clip (MP4) or 3D model (GLB) |
| `GET /media/{file}` | serve a generated PNG, MP4, GIF or GLB (`/images/{file}` still works) |

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
cd frontend && npm install && npm run build && cd ..   # once, and after UI changes
python server.py         # web UI (serves frontend/dist)  →  http://127.0.0.1:8000
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

### Working on the UI

Run the backend and the Vite dev server side by side; Vite hot-reloads the React app
and proxies `/api` and `/media` to the backend on :8000.

```bash
python server.py                 # terminal 1
cd frontend && npm run dev       # terminal 2  →  http://localhost:5173
```

## Models

All open-weights, served by the HF Inference API by default:

Models marked **free Space** also run on a public ZeroGPU Space, so they work even
without API access. The others are **API only**.

**Images**

- `Tongyi-MAI/Z-Image-Turbo` (default, free Space — 6B, 9 steps, excellent prompt
  adherence and legible text, Apache-2.0)
- `krea/Krea-2-Turbo` (free Space — June 2026, photographic look, 8 steps)
- `black-forest-labs/FLUX.2-klein-4B` (free Space — FLUX.2 distilled, 4 steps, ~10 s)
- `Qwen/Qwen-Image-2512` (API only — best for text in images; its public Space is
  currently broken upstream), `tencent/HunyuanImage-3.0`, `black-forest-labs/FLUX.1-dev`
  (API only)
- `black-forest-labs/FLUX.1-schnell` (free Space — the old default; fast but weakest)

**Video** (flip the switch)

- `MiniMaxAI/MiniMax-H3` (default, free Space — Aug 2026, the strongest open video
  model, 24 fps with a generated soundtrack; 2-14 s clips, a few minutes each on the
  full 28-step Space, with a 4-step mirror as fallback)
- `Wan-AI/Wan2.2-T2V-A14B`, `tencent/HunyuanVideo-1.5` (API only)
- `Lightricks/LTX-Video-0.9.7-distilled` (free Space — fast, ~20 s per clip, lower quality)

Text side: `Qwen/Qwen2.5-7B-Instruct` (swap via `TEXT_MODEL`).

Switch models in the sidebar or in `.env` (`IMAGE_MODEL`, `VIDEO_MODEL`, `VIDEO_DURATION`).

### About "free" video

Video is far more expensive than images. The Hugging Face Inference API bills video
against your token's monthly credit (a free account gets very little, so the Wan and
Hunyuan models effectively need a PRO plan or pay-as-you-go). The **LTX-Video public
Space** is genuinely free: it runs on Hugging Face's ZeroGPU hardware and needs no
token at all, but anonymous users get only a few clips per day and a free token raises
that quota a lot. Clips default to 2 seconds; the slider goes to 8, which takes
proportionally longer and eats more quota.

With `BACKEND=api` and a token, both tools try the Inference API first and fall back
to the Space on a permission or credit error. **Fine-grained tokens need the "Make
calls to Inference Providers" permission** (classic Read tokens have it); without it
every API call is a 403, the free Space is used instead, and the UI shows a one-time
notice. Models that the API only serves as image-editing or image-to-video tasks
(FLUX.2 klein, MiniMax-H3) always go to their Space.

## Running the diffusion model locally (no API calls)

```bash
pip install diffusers torch transformers accelerate
```

then set `BACKEND=local` in `.env`. First run downloads the weights (~10–20 GB for
FLUX); a GPU is strongly recommended — on CPU a single image takes several minutes.

## Notes

- Generated PNGs and MP4s are saved to `./generated/` and are downloadable from the chat.
- In demo mode (`BACKEND=demo`) a video turn writes an animated GIF placeholder instead
  of calling a model, so the whole UI can be exercised offline.
- The resolution control picks the **short side** of a video frame (e.g. 16:9 at 512 is
  896×512); frame sizes are snapped to multiples of 32 as LTX requires. The Inference
  API backend uses each provider's default frame size.
- **Enhance prompt** off in the sidebar sends your text verbatim to the diffusion model
  and skips the LLM entirely.
- If the text model is unreachable, the bot degrades gracefully: a keyword heuristic
  routes the turn and your raw prompt is used for generation.
- FLUX models ignore `negative_prompt` and use `guidance_scale=0`; SD models use both.
