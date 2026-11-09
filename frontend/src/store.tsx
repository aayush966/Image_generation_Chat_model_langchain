/* Studio state and actions, shared by every pane through React context.
 *
 * Two generation modes, "image" and "video". Each has its own model list and
 * defaults; prompt, memory and gallery are shared. The conversation lives
 * server-side in LangChain memory keyed by a session id; the browser keeps a
 * transcript copy only to redraw the stage after a reload and to re-seed the
 * server if it was restarted.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api } from "./lib/api";
import { getSize, isVideo, sessionMedia } from "./lib/media";
import { KEYS, getSessionId, readJSON, writeJSON } from "./lib/storage";
import type {
  AppConfig,
  ChatPayload,
  HistoryEntry,
  ModelDefaults,
  MediaItem,
  Mode,
  Pending,
  Reply,
  Settings,
  Toast,
  View,
} from "./lib/types";

/* Per-mode copy: everything that differs between drawing and filming. */
export const MODE_COPY = {
  image: {
    title: "What do you want to see?",
    text: "Describe a picture and an open-source diffusion model draws it. Follow-ups like “same but at night” keep the scene.",
    placeholder: "Describe an image…",
    button: "Generate",
    pending: ["Enhancing prompt, then drawing", "Drawing"],
    prompts: [
      "a red fox asleep in a snowy pine forest at dawn",
      "cyberpunk street market in the rain, neon reflections",
      "watercolour lighthouse during a storm",
      "a cosy reading nook with a cat, warm afternoon light",
    ],
  },
  video: {
    title: "What do you want to animate?",
    text: "Describe a short clip — what moves, and how the camera moves — and an open-source text-to-video model films it. Clips take a minute or two.",
    placeholder: "Describe a video clip…",
    button: "Generate video",
    pending: ["Enhancing prompt, then rendering", "Rendering video"],
    prompts: [
      "a red fox trotting through a snowy forest, slow tracking shot",
      "waves crashing on black rocks at golden hour, drone flyover",
      "steam rising from a cup of coffee on a rainy windowsill",
      "neon city street at night, cars passing, handheld camera",
    ],
  },
} as const;

export const BACKEND_STATUS: Record<string, { tone: "ok" | "warn" | "error"; label: string }> = {
  api: { tone: "ok", label: "Inference API" },
  space: { tone: "warn", label: "Public Space" },
  local: { tone: "ok", label: "Local GPU" },
  demo: { tone: "warn", label: "Demo mode" },
};

export interface ModelOption {
  id: string;
  label: string;
  badge: "Google" | "Free Space" | "API only" | "Needs token";
  disabled: boolean;
}

/* ───────────────────────── pure helpers ───────────────────────── */

const DEFAULT_SETTINGS: Settings = {
  mode: "image",
  model: null,
  videoModel: null,
  aspect: "1:1",
  res: "1024",
  videoRes: "512",
  steps: 4,
  guidance: 0,
  duration: 4,
  enhance: true,
  forceImage: true,
};

/** Saved settings from either UI generation (the old one stored numbers as strings). */
function loadSettings(params: URLSearchParams): { settings: Settings; hadSliders: boolean; hadDuration: boolean } {
  const saved = readJSON<Partial<Record<keyof Settings, unknown>> | null>(KEYS.settings, null) || {};
  const num = (v: unknown, fallback: number) => (v === undefined || v === null || v === "" ? fallback : Number(v));
  const urlMode = params.get("mode");
  const mode: Mode = (urlMode || saved.mode) === "video" ? "video" : "image";
  return {
    hadSliders: saved.steps !== undefined && saved.steps !== null,
    hadDuration: Boolean(saved.duration),
    settings: {
      mode,
      model: (saved.model as string) || null,
      videoModel: (saved.videoModel as string) || null,
      aspect: (["1:1", "16:9", "9:16"].includes(saved.aspect as string) ? saved.aspect : "1:1") as Settings["aspect"],
      res: String(saved.res || DEFAULT_SETTINGS.res),
      videoRes: String(saved.videoRes || DEFAULT_SETTINGS.videoRes),
      steps: num(saved.steps, DEFAULT_SETTINGS.steps),
      guidance: num(saved.guidance, DEFAULT_SETTINGS.guidance),
      duration: num(saved.duration, DEFAULT_SETTINGS.duration),
      enhance: typeof saved.enhance === "boolean" ? saved.enhance : true,
      forceImage: typeof saved.forceImage === "boolean" ? saved.forceImage : true,
    },
  };
}

export function modelOptions(cfg: AppConfig | null, mode: Mode): ModelOption[] {
  if (!cfg) return [];
  const video = mode === "video";
  const tokenFree = new Set(video ? cfg.token_free_video_models : cfg.token_free_models);
  const google = new Set(cfg.google_models || []);
  return (video ? cfg.video_models : cfg.models).map((id) => {
    // Without a token only the Space-backed (or Google) models can actually run.
    const needsToken = !cfg.has_token && !tokenFree.has(id) && !google.has(id);
    return {
      id,
      label: id.split("/")[1] || id,
      badge: google.has(id) ? "Google" : needsToken ? "Needs token" : tokenFree.has(id) ? "Free Space" : "API only",
      disabled: needsToken,
    };
  });
}

/** The model a mode will actually use: the saved pick if it can run, else the server default. */
export function resolveModel(cfg: AppConfig | null, s: Settings, mode: Mode): string | null {
  if (!cfg) return (mode === "video" ? s.videoModel : s.model) || null;
  const wanted = mode === "video" ? s.videoModel : s.model;
  const option = modelOptions(cfg, mode).find((o) => o.id === wanted);
  return option && !option.disabled ? option.id : mode === "video" ? cfg.video_model : cfg.image_model;
}

export function modelDefaults(
  cfg: AppConfig | null,
  mode: Mode,
  model: string | null
): (ModelDefaults & { fps?: number }) | undefined {
  if (!cfg || !model) return undefined;
  return mode === "video" ? cfg.video_defaults[model] : cfg.defaults[model];
}

/** Reset the step/guidance sliders to what the active model is tuned for. */
function withDefaults(cfg: AppConfig | null, s: Settings): Settings {
  const d = modelDefaults(cfg, s.mode, resolveModel(cfg, s, s.mode));
  return d ? { ...s, steps: d.steps, guidance: d.guidance_scale } : s;
}

/** Each mode remembers its own resolution; 1024-pixel video is too slow, so it caps at 768. */
export function resolutionFor(s: Settings, mode: Mode): number {
  const res = Number(mode === "video" ? s.videoRes : s.res);
  if (mode === "video" && res > 768) return 512;
  return res || (mode === "video" ? 512 : 1024);
}

/* ───────────────────────── the store ───────────────────────── */

function useStudioState() {
  const sessionId = useMemo(getSessionId, []);
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const boot = useMemo(() => loadSettings(params), [params]);

  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "offline">("loading");
  const [settings, setSettings] = useState<Settings>(boot.settings);
  const [seed, setSeed] = useState("");
  const [draft, setDraft] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>(() => readJSON<HistoryEntry[]>(KEYS.history, []));
  const [gallery, setGallery] = useState<MediaItem[]>([]);
  const [current, setCurrent] = useState<MediaItem | null>(() => {
    const items = sessionMedia(readJSON<HistoryEntry[]>(KEYS.history, []));
    return items[items.length - 1] ?? null;
  });
  const [remembered, setRemembered] = useState(() => readJSON<HistoryEntry[]>(KEYS.history, []).length);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null);
  const [replies, setReplies] = useState<Reply[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [view, setView] = useState<View>(() => {
    const v = params.get("view");
    return v === "gallery" || v === "3d" ? v : "create";
  });
  const [lightbox, setLightbox] = useState<MediaItem | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const warnedApiBlocked = useRef(false);
  const nextId = useRef(1);

  /* persistence */
  useEffect(() => writeJSON(KEYS.settings, settings), [settings]);
  useEffect(() => writeJSON(KEYS.history, history.slice(-24)), [history]);

  const mode = settings.mode;
  const model = resolveModel(cfg, settings, mode);
  const size = getSize(mode, settings.aspect, resolutionFor(settings, mode));

  /* ── notifications ── */
  const toast = useCallback((text: string, tone: Toast["tone"] = "ok") => {
    const id = nextId.current++;
    setToasts((t) => [...t, { id, tone, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const pushReply = useCallback((reply: Omit<Reply, "id"> & { retry?: () => void }) => {
    const id = nextId.current++;
    setReplies((r) => [...r, { ...reply, id } as Reply].slice(-4));
  }, []);

  const dismissReply = useCallback((id: number) => setReplies((r) => r.filter((x) => x.id !== id)), []);

  /* ── loading ── */
  const loadGallery = useCallback(async () => {
    try {
      setGallery(await api.gallery(48));
    } catch {
      /* the gallery is best-effort */
    }
  }, []);

  useEffect(() => {
    api
      .config()
      .then((c) => {
        setCfg(c);
        setStatus("ready");
        setSettings((s) => {
          let next = s;
          if (!boot.hadDuration && c.video?.duration) next = { ...next, duration: c.video.duration };
          return boot.hadSliders ? next : withDefaults(c, next);
        });
      })
      .catch(() => setStatus("offline"));
    api
      .session(sessionId)
      .then((d) => setRemembered(d.messages.length))
      .catch(() => {});
    loadGallery();
  }, [boot, sessionId, loadGallery]);

  /* ── settings actions ── */
  const updateSettings = useCallback((patch: Partial<Settings>) => setSettings((s) => ({ ...s, ...patch })), []);

  const setMode = useCallback(
    (next: Mode) => setSettings((s) => (s.mode === next ? s : withDefaults(cfg, { ...s, mode: next }))),
    [cfg]
  );

  const setModel = useCallback(
    (id: string) =>
      setSettings((s) => withDefaults(cfg, s.mode === "video" ? { ...s, videoModel: id } : { ...s, model: id })),
    [cfg]
  );

  const setResolution = useCallback(
    (res: string) => setSettings((s) => (s.mode === "video" ? { ...s, videoRes: res } : { ...s, res })),
    []
  );

  /* ── one conversation turn ── */
  const sendRef = useRef<(message: string, overrides?: Partial<ChatPayload>) => Promise<void>>(async () => {});

  const send = async (message: string, overrides: Partial<ChatPayload> = {}) => {
    if (busy || !message.trim()) return;
    setBusy(true);
    setView("create");

    const turnMode = overrides.mode ?? mode;
    const video = turnMode === "video";
    const turnModel = overrides.model ?? resolveModel(cfg, settings, turnMode);
    // A turn in the other mode (Vary on an older item) uses that model's tuned defaults.
    const d = turnMode !== mode ? modelDefaults(cfg, turnMode, turnModel) : undefined;
    const [width, height] = getSize(turnMode, settings.aspect, resolutionFor(settings, turnMode));
    const payload: ChatPayload = {
      message,
      session_id: sessionId,
      // Only read by the server if it has forgotten this session (e.g. a restart).
      history: history.map(({ role, content }) => ({ role, content })),
      mode: turnMode,
      model: turnModel,
      width,
      height,
      steps: d?.steps ?? settings.steps,
      guidance_scale: d?.guidance_scale ?? settings.guidance,
      seed: seed.trim() === "" ? null : Number(seed),
      duration: video ? settings.duration : null,
      enhance: settings.enhance,
      force_image: settings.forceImage,
      ...overrides,
    };

    const generating = payload.force_image || payload.enhance === false;
    const [enhancing, plain] = MODE_COPY[turnMode].pending;
    const started = performance.now();
    setPending({
      label: generating ? (payload.enhance ? enhancing : plain) : "Thinking",
      startedAt: started,
      width,
      height,
    });
    const retry = () => sendRef.current(message, overrides);

    try {
      const data = await api.chat(payload);
      const userTurn: HistoryEntry = { role: "user", content: message };
      if (data.type !== "text") {
        const item: MediaItem = {
          ...data,
          width,
          height,
          steps: payload.steps,
          seed: payload.seed,
          seconds: ((performance.now() - started) / 1000).toFixed(1),
        };
        // `content` stays byte-identical to what the server remembers.
        setHistory((h) => [
          ...h,
          userTurn,
          {
            role: "assistant",
            content: `${data.mode === "video" ? "Generated a video of" : "Generated an image of"}: ${data.prompt}`,
            media: item,
          },
        ]);
        setRemembered((r) => data.remembered ?? r);
        setCurrent(item);
        loadGallery();
        if (data.api_blocked && !warnedApiBlocked.current) {
          warnedApiBlocked.current = true;
          pushReply({
            kind: "notice",
            text:
              "Your Hugging Face token cannot call Inference Providers, so this was made on the free public Space " +
              "instead (same open-source model, smaller quota). To unlock the API-only models, enable “Make calls to " +
              "Inference Providers” on your token at https://huggingface.co/settings/tokens and restart the server.",
          });
        }
      } else {
        const degraded = /Inference Providers|without a Hugging Face token|couldn't reach the chat model/i;
        pushReply({ kind: degraded.test(data.text) ? "notice" : "reply", text: data.text });
        setHistory((h) => [...h, userTurn, { role: "assistant", content: data.text }]);
        setRemembered((r) => data.remembered ?? r);
      }
    } catch (err) {
      pushReply({ kind: "fail", text: err instanceof Error ? err.message : String(err), retry });
    } finally {
      setPending(null);
      setBusy(false);
    }
  };
  sendRef.current = send;

  /* ── stage actions ── */
  const copyPrompt = async () => {
    if (!current?.prompt) return;
    try {
      await navigator.clipboard.writeText(current.prompt);
      toast("Prompt copied to clipboard");
    } catch {
      toast("The browser blocked clipboard access", "error");
    }
  };

  /** Re-render the current prompt with a fresh seed, in the mode it was made in. */
  const vary = () => {
    if (!current?.prompt) return;
    const itemMode: Mode = isVideo(current) ? "video" : "image";
    if (itemMode !== mode) setMode(itemMode);
    setSeed("");
    send(current.prompt, { enhance: false, mode: itemMode, seed: null });
  };

  const newChat = async () => {
    // Clear both copies: the server's LangChain memory and the local transcript.
    try {
      await api.resetSession(sessionId);
    } catch {
      /* the local reset still clears what the user can see */
    }
    setHistory([]);
    setCurrent(null);
    setReplies([]);
    setRemembered(0);
    setView("create");
    toast("Started a new conversation");
  };

  const showItem = (item: MediaItem) => {
    setView("create");
    setCurrent(item);
  };

  return {
    sessionId,
    cfg,
    status,
    settings,
    mode,
    model,
    size,
    seed,
    setSeed,
    draft,
    setDraft,
    history,
    gallery,
    current,
    remembered,
    busy,
    pending,
    replies,
    toasts,
    view,
    setView,
    lightbox,
    setLightbox,
    sidebarOpen,
    setSidebarOpen,
    updateSettings,
    setMode,
    setModel,
    setResolution,
    send,
    copyPrompt,
    vary,
    newChat,
    showItem,
    loadGallery,
    dismissReply,
    toast,
  };
}

export type Studio = ReturnType<typeof useStudioState>;

const StudioContext = createContext<Studio | null>(null);

export function StudioProvider({ children }: { children: ReactNode }) {
  const studio = useStudioState();
  return <StudioContext.Provider value={studio}>{children}</StudioContext.Provider>;
}

export function useStudio(): Studio {
  const studio = useContext(StudioContext);
  if (!studio) throw new Error("useStudio must be used inside <StudioProvider>");
  return studio;
}
