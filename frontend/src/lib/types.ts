export type Mode = "image" | "video";
export type Aspect = "1:1" | "16:9" | "9:16";
export type View = "create" | "gallery" | "3d";
export type Theme = "dark" | "light";

export interface ModelDefaults {
  steps: number;
  guidance_scale: number;
}

export interface VideoModelDefaults extends ModelDefaults {
  fps: number;
}

/** GET /api/config */
export interface AppConfig {
  image_model: string;
  video_model: string;
  text_model: string;
  backend: "api" | "space" | "local" | "demo" | string;
  has_token: boolean;
  models: string[];
  video_models: string[];
  token_free_models: string[];
  token_free_video_models: string[];
  google_models: string[];
  memory_window: number;
  defaults: Record<string, ModelDefaults>;
  video_defaults: Record<string, VideoModelDefaults>;
  video: {
    duration: number;
    min_duration: number;
    max_duration: number;
    width: number;
    height: number;
  };
  api_blocked: string[];
}

/** A generated picture or clip, as the stage, rail and gallery show it. */
export interface MediaItem {
  type: Mode;
  url: string;
  filename: string;
  prompt?: string;
  model?: string;
  mode?: Mode;
  width?: number;
  height?: number;
  steps?: number;
  seed?: number | null;
  duration?: number | null;
  /** Wall-clock seconds the request took. */
  seconds?: string;
  /** Unix mtime, from /api/gallery. */
  created?: number;
}

/** One turn of the local transcript. Older transcripts stored media under `image`. */
export interface HistoryEntry {
  role: "user" | "assistant";
  content: string;
  media?: MediaItem;
  image?: MediaItem;
}

export interface ChatPayload {
  message: string;
  session_id: string;
  history: { role: string; content: string }[];
  mode: Mode;
  model: string | null;
  width: number;
  height: number;
  steps: number;
  guidance_scale: number;
  seed: number | null;
  duration: number | null;
  enhance: boolean;
  force_image: boolean;
}

export type ChatResponse =
  | {
      type: Mode;
      prompt: string;
      url: string;
      filename: string;
      model: string;
      mode: Mode;
      duration: number | null;
      remembered: number;
      api_blocked: boolean;
    }
  | { type: "text"; text: string; remembered: number };

/** Messages shown under the stage: bot replies, notices and failures. */
export type Reply =
  | { id: number; kind: "reply"; text: string }
  | { id: number; kind: "notice"; text: string }
  | { id: number; kind: "fail"; text: string; retry?: () => void };

export interface Toast {
  id: number;
  tone: "ok" | "error";
  text: string;
}

export interface Pending {
  label: string;
  startedAt: number;
  width: number;
  height: number;
}

/** What is persisted under `imagebot.settings` — same keys as the old UI. */
export interface Settings {
  mode: Mode;
  model: string | null;
  videoModel: string | null;
  aspect: Aspect;
  res: string;
  videoRes: string;
  steps: number;
  guidance: number;
  duration: number;
  enhance: boolean;
  forceImage: boolean;
}

/* ───────────── 3D Studio ───────────── */

export type Tool3D = "view" | "orbit" | "mesh";

/** POST /api/upload, and the photo a 3D Studio session works from. */
export interface Source3D {
  url: string;
  filename: string;
  width?: number;
  height?: number;
}

/** One render in 3D Studio: a new angle, an orbit clip or a 3D model. */
export interface Job3D {
  id: string;
  tool: Tool3D;
  /** "Left profile", "Orbit right", "3D model" … */
  label: string;
  /** Filename of the source photo this was made from. */
  source: string;
  status: "pending" | "done" | "error";
  startedAt: number;
  url?: string;
  filename?: string;
  error?: string;
}

/** POST /api/3d/{tool} */
export interface Result3D {
  type: "image" | "video" | "model";
  url: string;
  filename: string;
  tool: Tool3D;
}
