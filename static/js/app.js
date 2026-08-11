/* ImageBot front-end — talks to the FastAPI backend in server.py */

const $ = (id) => document.getElementById(id);

const el = {
  messages: $("messages"),
  welcome: $("welcome"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  model: $("model"),
  size: $("size"),
  steps: $("steps"),
  stepsVal: $("stepsVal"),
  guidance: $("guidance"),
  guidanceVal: $("guidanceVal"),
  seed: $("seed"),
  enhance: $("enhance"),
  forceImage: $("forceImage"),
  gallery: $("gallery"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  modelLabel: $("modelLabel"),
  lightbox: $("lightbox"),
  lightboxImg: $("lightboxImg"),
  sidebar: $("sidebar"),
  memoryText: $("memoryText"),
};

/**
 * The conversation itself is remembered server-side by LangChain, keyed by this
 * id. The browser keeps a transcript copy too, but only to redraw the chat after
 * a reload and to re-seed the server if it has been restarted.
 */
const sessionId = (() => {
  let id = localStorage.getItem("imagebot.session");
  if (!id) {
    id = (crypto.randomUUID?.() || String(Date.now() + Math.random())).replace(/-/g, "");
    localStorage.setItem("imagebot.session", id);
  }
  return id;
})();

let history = JSON.parse(localStorage.getItem("imagebot.history") || "[]");
let modelDefaults = {};
let memoryWindow = 16;
let busy = false;

/* ────────────────────────────── helpers ────────────────────────────── */

const saveHistory = () =>
  localStorage.setItem("imagebot.history", JSON.stringify(history.slice(-20)));

/** Show how much of the conversation LangChain is currently holding. */
function setMemoryCount(count) {
  if (count === undefined || count === null) return;
  el.memoryText.textContent = `🧠 memory: ${count}/${memoryWindow} messages`;
}

const scrollDown = () => {
  el.messages.scrollTop = el.messages.scrollHeight;
};

function bubble(role, node, { error = false } = {}) {
  el.welcome?.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🧑" : "🎨";

  const body = document.createElement("div");
  body.className = "bubble" + (error ? " error" : "");
  body.append(node);

  wrap.append(avatar, body);
  el.messages.append(wrap);
  scrollDown();
  return wrap;
}

function textNode(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function typingNode() {
  const t = document.createElement("div");
  t.className = "typing";
  t.innerHTML = "<i></i><i></i><i></i>";
  return t;
}

/** Card holding a generated image plus its prompt and actions. */
function imageCard(data) {
  const card = document.createElement("div");
  card.className = "card";

  const img = document.createElement("img");
  img.src = data.url;
  img.alt = data.prompt;
  img.loading = "lazy";
  img.addEventListener("click", () => openLightbox(data.url));

  const prompt = document.createElement("div");
  prompt.className = "prompt";
  prompt.textContent = data.prompt;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${data.model} · ${el.size.value} · ${el.steps.value} steps`;

  const actions = document.createElement("div");
  actions.className = "actions";

  const download = document.createElement("a");
  download.href = data.url;
  download.download = data.filename;
  download.textContent = "⬇ Download";

  const copy = document.createElement("button");
  copy.textContent = "⧉ Copy prompt";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(data.prompt);
    copy.textContent = "✓ Copied";
    setTimeout(() => (copy.textContent = "⧉ Copy prompt"), 1400);
  });

  const again = document.createElement("button");
  again.textContent = "↻ Variation";
  again.addEventListener("click", () => {
    el.seed.value = "";
    send(data.prompt, { enhance: false });
  });

  actions.append(download, copy, again);
  card.append(img, prompt, meta, actions);
  return card;
}

function openLightbox(url) {
  el.lightboxImg.src = url;
  el.lightbox.hidden = false;
}

/* ────────────────────────────── API ────────────────────────────── */

async function loadConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    modelDefaults = cfg.defaults || {};
    memoryWindow = cfg.memory_window || memoryWindow;

    el.model.innerHTML = "";
    const tokenFree = new Set(cfg.token_free_models || []);
    cfg.models.forEach((name) => {
      // Without a token only the Space-backed models can actually run.
      const needsToken = !cfg.has_token && !tokenFree.has(name);
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = (name.split("/")[1] || name) + (needsToken ? " — needs token" : "");
      opt.disabled = needsToken;
      opt.selected = name === cfg.image_model;
      el.model.append(opt);
    });
    applyModelDefaults();

    const states = {
      demo: ["warn", "demo mode — placeholder images"],
      space: ["ok", "ready · public HF Space (no token needed)"],
      local: ["ok", "ready · local diffusers"],
      api: cfg.has_token
        ? ["ok", "ready · HF Inference API"]
        : ["error", "no Hugging Face token in .env"],
    };
    const [tone, label] = states[cfg.backend] || ["warn", cfg.backend];
    el.statusDot.className = "dot " + tone;
    el.statusText.textContent = label;
  } catch (err) {
    el.statusDot.className = "dot error";
    el.statusText.textContent = "backend unreachable";
  }
}

async function loadGallery() {
  try {
    const { images } = await (await fetch("/api/gallery")).json();
    el.gallery.innerHTML = "";
    if (!images.length) {
      el.gallery.innerHTML = '<div class="empty">No images yet.</div>';
      return;
    }
    images.forEach((image) => {
      const thumb = document.createElement("img");
      thumb.src = image.url;
      thumb.alt = image.filename;
      thumb.loading = "lazy";
      thumb.title = image.filename;
      thumb.addEventListener("click", () => openLightbox(image.url));
      el.gallery.append(thumb);
    });
  } catch {
    /* gallery is best-effort */
  }
}

async function send(message, overrides = {}) {
  if (busy || !message.trim()) return;
  busy = true;
  el.send.disabled = true;

  bubble("user", textNode(message));
  const pending = bubble("bot", typingNode());

  const [width, height] = el.size.value.split("x").map(Number);
  const seed = el.seed.value.trim();

  const payload = {
    message,
    session_id: sessionId,
    // Only used if the server has forgotten this session (e.g. it restarted).
    history,
    model: el.model.value,
    width,
    height,
    steps: Number(el.steps.value),
    guidance_scale: Number(el.guidance.value),
    seed: seed === "" ? null : Number(seed),
    enhance: el.enhance.checked,
    force_image: el.forceImage.checked,
    ...overrides,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    pending.remove();

    setMemoryCount(data.remembered);

    if (!res.ok) {
      bubble("bot", textNode(`⚠ ${data.detail || res.statusText}`), { error: true });
    } else if (data.type === "image") {
      bubble("bot", imageCard(data));
      history.push(
        { role: "user", content: message },
        { role: "assistant", content: `Generated an image of: ${data.prompt}` }
      );
      loadGallery();
    } else {
      bubble("bot", textNode(data.text));
      history.push(
        { role: "user", content: message },
        { role: "assistant", content: data.text }
      );
    }
    saveHistory();
  } catch (err) {
    pending.remove();
    bubble("bot", textNode(`⚠ Network error: ${err.message}`), { error: true });
  } finally {
    busy = false;
    el.send.disabled = false;
    el.input.focus();
  }
}

/* ────────────────────────────── events ────────────────────────────── */

function applyModelDefaults() {
  const d = modelDefaults[el.model.value];
  if (!d) return;
  el.steps.value = d.steps;
  el.guidance.value = d.guidance_scale;
  el.stepsVal.textContent = d.steps;
  el.guidanceVal.textContent = Number(d.guidance_scale).toFixed(1);
  el.modelLabel.textContent = el.model.value;
}

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value;
  el.input.value = "";
  el.input.style.height = "auto";
  send(text);
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
});

el.model.addEventListener("change", applyModelDefaults);
el.steps.addEventListener("input", () => (el.stepsVal.textContent = el.steps.value));
el.guidance.addEventListener(
  "input",
  () => (el.guidanceVal.textContent = Number(el.guidance.value).toFixed(1))
);

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => send(chip.textContent))
);

$("clearChat").addEventListener("click", async () => {
  // Clear both copies: the server's LangChain memory and the local transcript.
  try {
    await fetch("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch {
    /* the reload still clears what the user can see */
  }
  history = [];
  saveHistory();
  location.reload();
});

$("refreshGallery").addEventListener("click", loadGallery);

$("themeToggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("imagebot.theme", next);
});

$("menuToggle").addEventListener("click", () => el.sidebar.classList.toggle("open"));

el.lightbox.addEventListener("click", () => (el.lightbox.hidden = true));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") el.lightbox.hidden = true;
});

/* ────────────────────────────── boot ────────────────────────────── */

document.documentElement.dataset.theme =
  localStorage.getItem("imagebot.theme") || "dark";

/** Redraw the transcript kept in localStorage, so a reload is not a blank page. */
function restoreTranscript() {
  history.forEach((turn) =>
    bubble(turn.role === "user" ? "user" : "bot", textNode(turn.content))
  );
}

async function loadMemory() {
  try {
    const data = await (await fetch(`/api/session/${sessionId}`)).json();
    setMemoryCount(data.messages.length);
  } catch {
    /* the count is cosmetic */
  }
}

restoreTranscript();
loadConfig().then(loadMemory);
loadGallery();
el.input.focus();
