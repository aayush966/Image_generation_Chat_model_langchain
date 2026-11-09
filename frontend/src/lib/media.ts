import type { Aspect, HistoryEntry, MediaItem, Mode } from "./types";

export const isVideo = (item?: Partial<MediaItem> | null) =>
  Boolean(item) && (item!.type === "video" || /\.(mp4|webm)$/i.test(item!.filename || item!.url || ""));

/** "1786186436-draw-a-lighthouse-in-a-storm.mp4" -> "draw a lighthouse in a storm" */
export const labelFromFilename = (filename = "") =>
  filename
    .replace(/\.(png|jpe?g|webp|gif|mp4|webm)$/i, "")
    .replace(/^\d+-/, "")
    .replace(/-/g, " ");

export const labelOf = (item: MediaItem) => item.prompt || labelFromFilename(item.filename);

/** "black-forest-labs/FLUX.1-dev" -> "FLUX.1-dev" */
export const shortModel = (id?: string | null) => (id ? id.split("/")[1] || id : "");

export function timeAgo(seconds: number) {
  const diff = Math.max(0, Date.now() / 1000 - seconds);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return `${Math.floor(diff / 86400)} d ago`;
}

/** Every picture or clip in the local transcript, oldest first. */
export const sessionMedia = (history: HistoryEntry[]) =>
  history.map((e) => e.media || e.image).filter((m): m is MediaItem => Boolean(m));

const snap32 = (v: number) => Math.min(1280, Math.max(256, Math.round(v / 32) * 32));

/** Aspect + resolution -> the width/height the backend wants.
 * Images sit on an 8px grid; video models want multiples of 32, and the
 * resolution picks the SHORT side so a 16:9 clip at "512" is 896×512. */
export function getSize(mode: Mode, aspect: Aspect, res: number): [number, number] {
  if (mode === "video") {
    const side = snap32(res);
    if (aspect === "1:1") return [side, side];
    const long = snap32((side * 16) / 9);
    return aspect === "16:9" ? [long, side] : [side, long];
  }
  if (aspect === "1:1") return [res, res];
  const short = Math.round((res * 9) / 16 / 8) * 8;
  return aspect === "16:9" ? [res, short] : [short, res];
}

export function newId(): string {
  return (globalThis.crypto?.randomUUID?.() || String(Date.now() + Math.random())).replace(/-/g, "");
}
