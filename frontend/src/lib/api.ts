import type { AppConfig, ChatPayload, ChatResponse, MediaItem, Result3D, Source3D, Tool3D } from "./types";

export class ApiError extends Error {}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.detail || res.statusText || `HTTP ${res.status}`);
  return body as T;
}

const post = (url: string, body: unknown) =>
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const api = {
  config: () => fetch("/api/config").then((r) => json<AppConfig>(r)),

  session: (id: string) =>
    fetch(`/api/session/${id}`).then((r) =>
      json<{ messages: { role: string; content: string }[]; window: number }>(r)
    ),

  resetSession: (id: string) => post("/api/session/reset", { session_id: id }),

  gallery: (limit = 48) =>
    fetch(`/api/gallery?limit=${limit}`)
      .then((r) => json<{ items?: MediaItem[]; images?: MediaItem[] }>(r))
      .then((d) => d.items || d.images || []),

  chat: (payload: ChatPayload) => post("/api/chat", payload).then((r) => json<ChatResponse>(r)),

  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return fetch("/api/upload", { method: "POST", body }).then((r) => json<Source3D>(r));
  },

  scene3d: (tool: Tool3D, body: { filename: string; angle?: string; move?: string; focus?: string }) =>
    post(`/api/3d/${tool}`, body).then((r) => json<Result3D>(r)),
};
