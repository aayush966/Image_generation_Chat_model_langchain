/* 3D Studio state: the source photo and every render made from it.
 *
 * Kept in its own context so the Create stage, the lightbox and the gallery can
 * all hand an image over ("Open in 3D Studio"). The source and finished renders
 * are saved to localStorage; they are just server URLs.
 */

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "./lib/api";
import { newId } from "./lib/media";
import { readJSON, writeJSON } from "./lib/storage";
import type { Job3D, Source3D, Tool3D } from "./lib/types";

const KEY = "imagebot.studio3d";

export const ANGLES = [
  { id: "left", label: "Left 45°" },
  { id: "right", label: "Right 45°" },
  { id: "left-profile", label: "Left side" },
  { id: "right-profile", label: "Right side" },
  { id: "front", label: "Front" },
  { id: "behind", label: "Behind" },
  { id: "above", label: "From above" },
  { id: "below", label: "Low angle" },
] as const;

export const MOVES = [
  { id: "orbit-left", label: "Orbit left" },
  { id: "orbit-right", label: "Orbit right" },
  { id: "push-in", label: "Push in" },
  { id: "crane-up", label: "Crane up" },
] as const;

interface Saved {
  source: Source3D | null;
  jobs: Job3D[];
}

function useScene3DState() {
  const [saved] = useState(() => readJSON<Saved>(KEY, { source: null, jobs: [] }));
  const [source, setSourceState] = useState<Source3D | null>(saved.source);
  // Pending jobs do not survive a reload: the request died with the page.
  const [jobs, setJobs] = useState<Job3D[]>(() => (saved.jobs || []).filter((j) => j.status === "done"));
  // null = the original photo is on the viewer.
  const [selected, setSelected] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const sourceRef = useRef(source);
  sourceRef.current = source;

  useEffect(() => {
    writeJSON(KEY, { source, jobs: jobs.filter((j) => j.status === "done").slice(-40) });
  }, [source, jobs]);

  const setSource = useCallback((next: Source3D | null) => {
    setSourceState(next);
    setSelected(null);
    setUploadError(null);
  }, []);

  const upload = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setUploadError("Choose an image file (JPG, PNG or WebP).");
        return;
      }
      setUploading(true);
      setUploadError(null);
      try {
        setSource(await api.upload(file));
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : String(err));
      } finally {
        setUploading(false);
      }
    },
    [setSource]
  );

  const run = useCallback(async (tool: Tool3D, label: string, opts: { angle?: string; move?: string; focus?: string }) => {
    const src = sourceRef.current;
    if (!src) return;
    const id = newId();
    const job: Job3D = { id, tool, label, source: src.filename, status: "pending", startedAt: Date.now() };
    setJobs((j) => [...j, job]);
    setSelected(id);
    try {
      const res = await api.scene3d(tool, { filename: src.filename, ...opts });
      setJobs((j) => j.map((x) => (x.id === id ? { ...x, status: "done", url: res.url, filename: res.filename } : x)));
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      setJobs((j) => j.map((x) => (x.id === id ? { ...x, status: "error", error } : x)));
    }
  }, []);

  const remove = useCallback((id: string) => {
    setJobs((j) => j.filter((x) => x.id !== id));
    setSelected((s) => (s === id ? null : s));
  }, []);

  const current = source ? jobs.filter((j) => j.source === source.filename) : [];

  return {
    source,
    setSource,
    upload,
    uploading,
    uploadError,
    jobs: current,
    selected,
    setSelected,
    run,
    remove,
  };
}

export type Scene3D = ReturnType<typeof useScene3DState>;

const Scene3DContext = createContext<Scene3D | null>(null);

export function Scene3DProvider({ children }: { children: ReactNode }) {
  const value = useScene3DState();
  return <Scene3DContext.Provider value={value}>{children}</Scene3DContext.Provider>;
}

export function useScene3D(): Scene3D {
  const value = useContext(Scene3DContext);
  if (!value) throw new Error("useScene3D must be used inside <Scene3DProvider>");
  return value;
}
