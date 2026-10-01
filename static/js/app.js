/* ImageBot Studio front-end — talks to the FastAPI backend in server.py.
 *
 * The conversation is remembered server-side by LangChain, keyed by a session
 * id. The browser keeps a transcript copy too, but only to redraw the stage
 * after a reload and to re-seed the server if it has been restarted.
 */

const $ = (id) => document.getElementById(id);

const el = {
  app: $("app"),
  side: $("side"),
  scrim: $("scrim"),
  frame: $("frame"),
  hero: $("hero"),
  stageImg: $("stageImg"),
  stageExpand: $("stageExpand"),
  pending: $("pending"),
  pendingLabel: $("pendingLabel"),
  pendingElapsed: $("pendingElapsed"),
  stageBar: $("stageBar"),
  caption: $("caption"),
  captionPrompt: $("captionPrompt"),
  captionTags: $("captionTags"),
  replies: $("replies"),
  likeCount: $("likeCount"),
  sessionCount: $("sessionCount"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  model: $("model"),
  steps: $("steps"),
  stepsVal: $("stepsVal"),
  stepsNote: $("stepsNote"),
  guidance: $("guidance"),
  guidanceVal: $("guidanceVal"),
  guidanceNote: $("guidanceNote"),
  guidanceField: $("guidanceField"),
  sizeVal: $("sizeVal"),
  qualityLabel: $("qualityLabel"),
  seed: $("seed"),
  enhanceTool: $("enhanceTool"),
  forceImage: $("forceImage"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  statusModel: $("statusModel"),
  statusBadge: $("statusBadge"),
  popMeta: $("popMeta"),
  galleryGrid: $("galleryGrid"),
  galleryCount: $("galleryCount"),
  galleryMeta: $("galleryMeta"),
  themeIcon: $("themeIcon"),
  recentCards: $("recentCards"),
  memWindow: $("memWindow"),
  memFill: $("memFill"),
  memCount: $("memCount"),
  sessionLabel: $("sessionLabel"),
  lightbox: $("lightbox"),
  lightboxImg: $("lightboxImg"),
  lightboxCaption: $("lightboxCaption"),
  lightboxDownload: $("lightboxDownload"),
  toasts: $("toasts"),
};

const KEYS = {
  session: "imagebot.session",
  history: "imagebot.history",
  theme: "imagebot.theme",
  settings: "imagebot.settings",
  liked: "imagebot.liked",
};

function readJSON(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}

const sessionId = (() => {
  let id = localStorage.getItem(KEYS.session);
  if (!id) {
    id = (crypto.randomUUID?.() || String(Date.now() + Math.random())).replace(/-/g, "");
    localStorage.setItem(KEYS.session, id);
  }
  return id;
})();

const state = {
  history: readJSON(KEYS.history, []),
  liked: readJSON(KEYS.liked, {}),
  gallery: [],
  current: null, // the picture on the stage
  recentMode: "session",
  remembered: 0,
  defaults: {},
  memoryWindow: 16,
  busy: false,
  enhance: true,
  pendingTimer: null,
};

/* ────────────────────────────── primitives ────────────────────────────── */

function icon(name, className = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function tag(text) {
  const node = document.createElement("span");
  node.className = "tag";
  node.textContent = text;
  return node;
}

const saveHistory = () =>
  localStorage.setItem(KEYS.history, JSON.stringify(state.history.slice(-24)));

const saveLiked = () => localStorage.setItem(KEYS.liked, JSON.stringify(state.liked));

function toast(message, tone = "ok") {
  const node = document.createElement("div");
  node.className = `toast ${tone}`;
  node.append(icon(tone === "error" ? "alert" : "check"), document.createTextNode(message));
  el.toasts.append(node);
  setTimeout(() => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 220);
  }, 2200);
}

const sessionImages = () => state.history.filter((entry) => entry.image).map((entry) => entry.image);

/** "1786186436-draw-a-lighthouse-in-a-storm.png" -> "draw a lighthouse in a storm" */
function labelFromFilename(filename) {
  return (filename || "")
    .replace(/\.png$/i, "")
    .replace(/^\d+-/, "")
    .replace(/-/g, " ");
}

function timeAgo(seconds) {
  const diff = Math.max(0, Date.now() / 1000 - seconds);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return `${Math.floor(diff / 86400)} d ago`;
}

/* ─────────────────────────── the stage ─────────────────────────── */

/** Put a picture on the stage and wire the action bar to it. */
function showResult(data) {
  state.current = data;
  el.hero.hidden = true;

  el.frame.classList.toggle("is-wide", data.width > data.height);
  el.frame.classList.toggle("is-tall", data.height > data.width);

  el.stageImg.classList.remove("loaded");
  el.stageImg.hidden = false;
  el.stageImg.src = data.url;
  el.stageImg.alt = data.prompt || "";
  el.stageExpand.hidden = false;

  el.stageBar.hidden = false;
  el.caption.hidden = false;
  el.captionPrompt.textContent = data.prompt || labelFromFilename(data.filename);
  el.captionPrompt.classList.remove("open");

  const shortModel = (data.model || "").split("/")[1] || data.model;
  el.captionTags.replaceChildren(document.createTextNode("Tags"));
  if (shortModel) el.captionTags.append(tag(shortModel));
  if (data.width) el.captionTags.append(tag(`${data.width}×${data.height}`));
  if (data.steps) el.captionTags.append(tag(`${data.steps} steps`));
  if (data.seed !== null && data.seed !== undefined) el.captionTags.append(tag(`seed ${data.seed}`));
  if (data.seconds) el.captionTags.append(tag(`${data.seconds}s`));

  $("actDownload").href = data.url;
  $("actDownload").download = data.filename || "image.png";
  syncLike();
  renderRecent();
}

function syncLike() {
  const liked = Boolean(state.current && state.liked[state.current.filename]);
  $("actLike").setAttribute("aria-pressed", String(liked));
  el.likeCount.textContent = Object.keys(state.liked).length;
  el.sessionCount.textContent = sessionImages().length;
}

/** Spinner over the stage while the model runs; keeps the final aspect ratio. */
function startPending(width, height, label) {
  el.frame.classList.toggle("is-wide", width > height);
  el.frame.classList.toggle("is-tall", height > width);
  el.pendingLabel.textContent = label;
  el.pendingElapsed.textContent = "0.0s";
  el.pending.hidden = false;
  const started = performance.now();
  clearInterval(state.pendingTimer);
  state.pendingTimer = setInterval(() => {
    el.pendingElapsed.textContent = ((performance.now() - started) / 1000).toFixed(1) + "s";
  }, 100);
}

function stopPending() {
  clearInterval(state.pendingTimer);
  el.pending.hidden = true;
  if (state.current) {
    el.frame.classList.toggle("is-wide", state.current.width > state.current.height);
    el.frame.classList.toggle("is-tall", state.current.height > state.current.width);
  }
}

/* ─────────────────────────── replies ─────────────────────────── */

function pushReply(node) {
  el.replies.append(node);
  while (el.replies.children.length > 4) el.replies.firstElementChild.remove();
  node.scrollIntoView({ block: "nearest" });
}

function replyNode(text) {
  const node = document.createElement("div");
  node.className = "reply";
  node.textContent = text;
  return node;
}

/** A degraded-backend message, rendered as a notice rather than as speech. */
function noticeNode(text) {
  const box = document.createElement("div");
  box.className = "notice";
  const body = document.createElement("div");
  const url = text.match(/https?:\/\/\S+?(?=[)\s]|$)/);
  if (url) {
    body.append(document.createTextNode(text.slice(0, url.index)));
    const link = document.createElement("a");
    link.href = url[0];
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = url[0];
    body.append(link, document.createTextNode(text.slice(url.index + url[0].length)));
  } else {
    body.textContent = text;
  }
  box.append(icon("alert"), body);
  return box;
}

function failNode(detail, retry) {
  const box = document.createElement("div");
  box.className = "fail";

  const head = document.createElement("div");
  head.className = "fail-head";
  const body = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = "That didn't work";
  const code = document.createElement("code");
  code.textContent = detail;
  body.append(title, code);
  head.append(icon("alert"), body);
  box.append(head);

  if (retry) {
    const again = document.createElement("button");
    again.className = "btn";
    again.type = "button";
    again.append(icon("repeat"), document.createTextNode("Try again"));
    again.addEventListener("click", () => {
      box.remove();
      retry();
    });
    box.append(again);
  }
  return box;
}

/* ────────────────────────────── lightbox ────────────────────────────── */

function openLightbox(data) {
  el.lightboxImg.src = data.url;
  el.lightboxImg.alt = data.prompt || "";
  el.lightboxCaption.textContent = data.prompt || labelFromFilename(data.filename);
  el.lightboxDownload.href = data.url;
  el.lightboxDownload.download = data.filename || "image.png";
  el.lightbox.hidden = false;
}

const closeLightbox = () => {
  el.lightbox.hidden = true;
  el.lightboxImg.src = "";
};

/* ──────────────────────────── settings ──────────────────────────── */

const radio = (name) => document.querySelector(`input[name="${name}"]:checked`);

/** Aspect + resolution -> the width/height the backend wants, on an 8px grid. */
function getSize() {
  const res = Number(radio("res")?.value || 1024);
  const aspect = radio("aspect")?.value || "1:1";
  if (aspect === "1:1") return [res, res];
  const short = Math.round((res * 9) / 16 / 8) * 8;
  return aspect === "16:9" ? [res, short] : [short, res];
}

function refreshLabels() {
  const [w, h] = getSize();
  el.sizeVal.textContent = `${w} × ${h}`;
  el.qualityLabel.textContent = `${w}×${h} · ${el.steps.value} steps`;
}

function saveSettings() {
  localStorage.setItem(
    KEYS.settings,
    JSON.stringify({
      model: el.model.value,
      aspect: radio("aspect")?.value,
      res: radio("res")?.value,
      steps: el.steps.value,
      guidance: el.guidance.value,
      enhance: state.enhance,
      forceImage: el.forceImage.checked,
    })
  );
}

function restoreSettings() {
  const s = readJSON(KEYS.settings, null);
  if (!s) return null;
  const pick = (name, value) => {
    const input = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (input) input.checked = true;
  };
  if (s.aspect) pick("aspect", s.aspect);
  if (s.res) pick("res", s.res);
  if (typeof s.forceImage === "boolean") el.forceImage.checked = s.forceImage;
  if (typeof s.enhance === "boolean") setEnhance(s.enhance);
  return s;
}

function setEnhance(on) {
  state.enhance = on;
  el.enhanceTool.setAttribute("aria-pressed", String(on));
}

/** Per-model step/guidance defaults from /api/config, plus the FLUX caveat. */
function applyModelDefaults({ keepSliders = false } = {}) {
  const model = el.model.value;
  const defaults = state.defaults[model];

  if (defaults && !keepSliders) {
    el.steps.value = defaults.steps;
    el.guidance.value = defaults.guidance_scale;
  }
  el.stepsVal.textContent = el.steps.value;
  el.guidanceVal.textContent = Number(el.guidance.value).toFixed(1);

  const isFlux = /flux/i.test(model);
  el.guidanceField.classList.toggle("is-muted", isFlux);
  el.guidanceNote.textContent = isFlux ? "FLUX ignores guidance — it always runs at 0." : "";
  el.stepsNote.textContent = defaults
    ? `${model.split("/")[1] || model} is tuned for ${defaults.steps} steps.`
    : "";
  el.statusModel.textContent = model.split("/")[1] || model;
  refreshLabels();
}

/* ────────────────────────────── memory meter ────────────────────────────── */

function renderMemory() {
  const used = Math.min(state.remembered, state.memoryWindow);
  el.memWindow.textContent = state.memoryWindow;
  el.memCount.textContent = `${used} / ${state.memoryWindow}`;
  el.memFill.style.width = `${(used / state.memoryWindow) * 100}%`;
  el.sessionLabel.textContent = `session ${sessionId.slice(0, 6)}`;
}

/* ────────────────────────────── API ────────────────────────────── */

async function loadConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    state.defaults = cfg.defaults || {};
    state.memoryWindow = cfg.memory_window || state.memoryWindow;

    const saved = restoreSettings();
    const tokenFree = new Set(cfg.token_free_models || []);

    el.model.replaceChildren();
    cfg.models.forEach((name) => {
      // Without a token only the Space-backed models can actually run.
      const needsToken = !cfg.has_token && !tokenFree.has(name);
      const option = document.createElement("option");
      option.value = name;
      option.textContent = (name.split("/")[1] || name) + (needsToken ? " · needs token" : "");
      option.disabled = needsToken;
      el.model.append(option);
    });
    el.model.value =
      saved?.model && cfg.models.includes(saved.model) ? saved.model : cfg.image_model;
    if (el.model.selectedOptions[0]?.disabled) el.model.value = cfg.image_model;

    if (saved?.steps) el.steps.value = saved.steps;
    if (saved?.guidance) el.guidance.value = saved.guidance;
    applyModelDefaults({ keepSliders: Boolean(saved?.steps) });

    const states = {
      api: cfg.has_token ? ["ok", "Inference API"] : ["error", "no token"],
      space: ["warn", "public Space"],
      local: ["ok", "local GPU"],
      demo: ["warn", "demo mode"],
    };
    const [tone, label] = states[cfg.backend] || ["warn", cfg.backend];
    el.statusBadge.className = `status-badge ${tone}`;
    el.statusBadge.lastChild.textContent = cfg.has_token ? "Token" : "Free";
    el.statusText.textContent = label;

    el.popMeta.replaceChildren(
      tag(`backend ${cfg.backend}`),
      tag(cfg.has_token ? "token ✓" : "no token"),
      tag((cfg.text_model || "").split("/")[1] || "text model"),
      tag(`memory ${cfg.memory_window}`)
    );
    renderMemory();
  } catch {
    el.statusBadge.className = "status-badge error";
    el.statusText.textContent = "offline";
    el.statusModel.textContent = "server not reachable";
  }
}

async function loadSession() {
  try {
    const data = await (await fetch(`/api/session/${sessionId}`)).json();
    state.remembered = data.messages.length;
    renderMemory();
  } catch {
    /* best-effort */
  }
}

async function loadGallery() {
  try {
    const { images } = await (await fetch("/api/gallery?limit=48")).json();
    state.gallery = images;
    el.galleryCount.textContent = images.length;
    el.galleryMeta.textContent = images.length
      ? `${images.length} image${images.length === 1 ? "" : "s"}, newest first.`
      : "Nothing generated yet.";

    el.galleryGrid.replaceChildren();
    if (!images.length) {
      const empty = document.createElement("div");
      empty.className = "gallery-empty";
      empty.textContent = "Your generated images will collect here.";
      el.galleryGrid.append(empty);
    } else {
      images.forEach((image) => {
        const cell = document.createElement("figure");
        cell.className = "gallery-cell";
        const img = document.createElement("img");
        img.src = image.url;
        img.alt = labelFromFilename(image.filename);
        img.loading = "lazy";
        const caption = document.createElement("figcaption");
        caption.textContent = labelFromFilename(image.filename);
        cell.append(img, caption);
        cell.addEventListener("click", () => openLightbox(image));
        el.galleryGrid.append(cell);
      });
    }
    renderRecent();
  } catch {
    /* the gallery is best-effort */
  }
}

/* ────────────────────────────── recent rail ────────────────────────────── */

function recentCard(item, index) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "card";
  if (state.current && item.filename === state.current.filename) card.classList.add("is-current");

  const label = item.prompt || labelFromFilename(item.filename);

  const title = document.createElement("div");
  title.className = "card-title";
  const n = document.createElement("span");
  n.className = "n";
  n.textContent = `#${index + 1}`;
  const t = document.createElement("span");
  t.textContent = label.length > 32 ? label.slice(0, 32) + "…" : label;
  title.append(n, t);

  const prompt = document.createElement("div");
  prompt.className = "card-prompt";
  prompt.append(document.createTextNode("Prompt"));
  const p = document.createElement("span");
  p.textContent = label;
  prompt.append(p);

  const thumb = document.createElement("div");
  thumb.className = "card-thumb";
  const img = document.createElement("img");
  img.src = item.url;
  img.alt = label;
  img.loading = "lazy";
  thumb.append(img);

  const foot = document.createElement("div");
  foot.className = "card-foot";
  const when = item.created ? timeAgo(item.created) : "this session";
  foot.append(icon("clock"), document.createTextNode(when));
  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = (item.model || "").split("/")[1] || (state.liked[item.filename] ? "kept" : "");
  foot.append(rank);

  card.append(title, prompt, thumb, foot);
  card.addEventListener("click", () => {
    showView("create");
    showResult({ ...item, prompt: label });
  });
  return card;
}

function renderRecent() {
  const items = state.recentMode === "session" ? [...sessionImages()].reverse() : state.gallery;
  el.recentCards.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "rail-empty";
    empty.textContent =
      state.recentMode === "session"
        ? "Your generations will collect here."
        : "Nothing generated yet.";
    el.recentCards.append(empty);
    return;
  }
  items.slice(0, 3).forEach((item, i) => el.recentCards.append(recentCard(item, i)));
  if (items.length > 3) {
    const more = document.createElement("button");
    more.type = "button";
    more.className = "rail-more";
    more.textContent = `Show all ${state.recentMode === "session" ? sessionImages().length : state.gallery.length} in the gallery`;
    more.addEventListener("click", () => showView("gallery"));
    el.recentCards.append(more);
  }
}

/* ────────────────────────────── one turn ────────────────────────────── */

async function send(message, overrides = {}) {
  if (state.busy || !message.trim()) return;
  setBusy(true);
  showView("create");

  const [width, height] = getSize();
  const seedText = el.seed.value.trim();
  const payload = {
    message,
    session_id: sessionId,
    // Only read by the server if it has forgotten this session (e.g. a restart).
    history: state.history.map(({ role, content }) => ({ role, content })),
    model: el.model.value,
    width,
    height,
    steps: Number(el.steps.value),
    guidance_scale: Number(el.guidance.value),
    seed: seedText === "" ? null : Number(seedText),
    enhance: state.enhance,
    force_image: el.forceImage.checked,
    ...overrides,
  };

  state.history.push({ role: "user", content: message });

  const drawing = payload.force_image || payload.enhance === false;
  el.hero.hidden = true;
  startPending(width, height, drawing ? (payload.enhance ? "Enhancing, then drawing" : "Drawing") : "Thinking");
  const started = performance.now();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    stopPending();

    if (!res.ok) {
      state.history.pop(); // the turn never landed, so don't re-seed it
      if (!state.current) el.hero.hidden = false;
      pushReply(failNode(data.detail || res.statusText, () => send(message, overrides)));
    } else if (data.type === "image") {
      const card = {
        ...data,
        width,
        height,
        steps: payload.steps,
        seed: payload.seed,
        seconds: ((performance.now() - started) / 1000).toFixed(1),
      };
      // Keep the picture in the local transcript so a reload redraws it, while
      // `content` stays byte-identical to what the server remembers.
      state.history.push({
        role: "assistant",
        content: `Generated an image of: ${data.prompt}`,
        image: card,
      });
      state.remembered = data.remembered ?? state.remembered;
      showResult(card);
      renderMemory();
      loadGallery();
    } else {
      if (!state.current) el.hero.hidden = false;
      const degraded = /Inference Providers|without a Hugging Face token|couldn't reach the chat model/i;
      pushReply(degraded.test(data.text) ? noticeNode(data.text) : replyNode(data.text));
      state.history.push({ role: "assistant", content: data.text });
      state.remembered = data.remembered ?? state.remembered;
      renderMemory();
    }
    saveHistory();
  } catch (err) {
    stopPending();
    state.history.pop();
    if (!state.current) el.hero.hidden = false;
    pushReply(failNode(`Network error: ${err.message}`, () => send(message, overrides)));
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  state.busy = busy;
  el.send.disabled = busy;
  el.send.classList.toggle("busy", busy);
  if (!busy) el.input.focus();
}

/* ────────────────────────────── views ────────────────────────────── */

function showView(name) {
  $("viewCreate").hidden = name !== "create";
  $("viewGallery").hidden = name !== "gallery";
  document.querySelectorAll(".nav-item[data-view]").forEach((item) =>
    item.classList.toggle("is-active", item.dataset.view === name)
  );
  if (name === "gallery") loadGallery();
  closeSidebar();
}

function openSidebar() {
  el.side.classList.add("is-open");
  el.scrim.hidden = false;
  $("menuToggle").setAttribute("aria-expanded", "true");
}
function closeSidebar() {
  el.side.classList.remove("is-open");
  el.scrim.hidden = true;
  $("menuToggle").setAttribute("aria-expanded", "false");
}

/* ────────────────────────────── events ────────────────────────────── */

function autogrow() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 190) + "px";
}

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value;
  el.input.value = "";
  autogrow();
  send(text);
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});
el.input.addEventListener("input", autogrow);

el.model.addEventListener("change", () => {
  applyModelDefaults();
  saveSettings();
});

el.steps.addEventListener("input", () => {
  el.stepsVal.textContent = el.steps.value;
  refreshLabels();
});
el.guidance.addEventListener("input", () => {
  el.guidanceVal.textContent = Number(el.guidance.value).toFixed(1);
});

document.querySelectorAll('input[name="aspect"], input[name="res"]').forEach((input) =>
  input.addEventListener("change", () => {
    refreshLabels();
    saveSettings();
  })
);

[el.steps, el.guidance, el.forceImage].forEach((input) =>
  input.addEventListener("change", saveSettings)
);

el.enhanceTool.addEventListener("click", () => {
  setEnhance(!state.enhance);
  saveSettings();
});

$("rollSeed").addEventListener("click", () => {
  el.seed.value = Math.floor(Math.random() * 2 ** 31);
});

document.querySelectorAll(".hero-prompt").forEach((button) =>
  button.addEventListener("click", () => send(button.textContent.trim()))
);

/* style presets append a suffix to whatever is in the prompt */
document.querySelectorAll(".chat-item[data-style]").forEach((button) =>
  button.addEventListener("click", () => {
    const base = el.input.value.trim();
    const suffix = button.dataset.style;
    el.input.value = base ? `${base}, ${suffix}` : suffix;
    autogrow();
    el.input.focus();
    closeSidebar();
  })
);

/* stage actions */
$("actLike").addEventListener("click", () => {
  if (!state.current) return;
  const key = state.current.filename;
  if (state.liked[key]) delete state.liked[key];
  else state.liked[key] = true;
  saveLiked();
  syncLike();
  renderRecent();
});
$("actPreview").addEventListener("click", () => state.current && openLightbox(state.current));
el.stageExpand.addEventListener("click", () => state.current && openLightbox(state.current));
el.stageImg.addEventListener("click", () => state.current && openLightbox(state.current));
el.stageImg.addEventListener("load", () => el.stageImg.classList.add("loaded"));
$("actVary").addEventListener("click", () => {
  if (!state.current) return;
  el.seed.value = "";
  send(state.current.prompt, { enhance: false });
});
$("actCopy").addEventListener("click", async () => {
  if (!state.current) return;
  try {
    await navigator.clipboard.writeText(state.current.prompt);
    toast("Prompt copied");
  } catch {
    toast("Clipboard blocked by the browser", "error");
  }
});
el.captionPrompt.addEventListener("click", () => el.captionPrompt.classList.toggle("open"));

/* navigation */
document.querySelectorAll(".nav-item[data-view]").forEach((item) =>
  item.addEventListener("click", () => showView(item.dataset.view))
);
$("navSettings").addEventListener("click", () => {
  const panel = $("settingsPanel");
  panel.hidden = !panel.hidden;
  $("navSettings").setAttribute("aria-expanded", String(!panel.hidden));
});
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === ",") {
    event.preventDefault();
    $("navSettings").click();
  }
});
$("refreshGallery").addEventListener("click", loadGallery);
$("railRefresh").addEventListener("click", loadGallery);

document.querySelectorAll(".toggle [data-recent]").forEach((button) =>
  button.addEventListener("click", () => {
    state.recentMode = button.dataset.recent;
    document.querySelectorAll(".toggle [data-recent]").forEach((other) =>
      other.setAttribute("aria-pressed", String(other === button))
    );
    renderRecent();
  })
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
  state.history = [];
  saveHistory();
  location.reload();
});

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem(KEYS.theme, next);
  syncThemeIcon();
}
$("themeToggle").addEventListener("click", toggleTheme);
$("themeToggleMobile").addEventListener("click", toggleTheme);

function syncThemeIcon() {
  const dark = document.documentElement.dataset.theme === "dark";
  el.themeIcon.firstElementChild.setAttribute("href", dark ? "#i-sun" : "#i-moon");
  $("themeToggleMobile").firstElementChild.firstElementChild.setAttribute("href", dark ? "#i-sun" : "#i-moon");
}

/* mobile sidebar */
$("menuToggle").addEventListener("click", () =>
  el.side.classList.contains("is-open") ? closeSidebar() : openSidebar()
);
el.scrim.addEventListener("click", closeSidebar);

/* lightbox */
$("lightboxClose").addEventListener("click", closeLightbox);
el.lightbox.addEventListener("click", (event) => {
  if (event.target === el.lightbox) closeLightbox();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!el.lightbox.hidden) closeLightbox();
  else closeSidebar();
});

/* ────────────────────────────── boot ────────────────────────────── */

/** Put the last picture from localStorage back on the stage after a reload. */
function restoreTranscript() {
  const images = sessionImages();
  if (images.length) showResult(images[images.length - 1]);
  state.remembered = state.history.length;
  renderMemory();
  renderRecent();
}

syncThemeIcon();
restoreTranscript();
loadConfig();
loadSession();
loadGallery();
el.input.focus();
