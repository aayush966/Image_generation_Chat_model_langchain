import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Download, Rotate3d, Wand2, X } from "lucide-react";
import { useEffect } from "react";
import { isVideo, labelOf } from "../lib/media";
import { useScene3D } from "../scene3d";
import { useStudio } from "../store";
import { IconButton } from "./ui";

export function Lightbox() {
  const s = useStudio();
  const d = useScene3D();
  const item = s.lightbox;

  useEffect(() => {
    if (!item) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [item]);

  if (!item) return null;
  const video = isVideo(item);
  const label = labelOf(item);
  const close = () => s.setLightbox(null);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Preview"
      className="fixed inset-0 z-50 flex flex-col bg-black/90 backdrop-blur-sm animate-fade-up"
      onClick={(e) => e.target === e.currentTarget && close()}
    >
      <div className="flex shrink-0 items-center gap-3 px-4 py-3 text-white">
        <p className="min-w-0 flex-1 truncate text-[13px] text-white/80">{label}</p>
        <button
          type="button"
          onClick={() => {
            s.showItem({ ...item, prompt: label });
            close();
          }}
          className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[12.5px] font-medium text-white/80 hover:bg-white/10 hover:text-white"
        >
          <Wand2 className="size-4" /> Open in studio
        </button>
        {!video && (
          <button
            type="button"
            onClick={() => {
              d.setSource({ url: item.url, filename: item.filename });
              s.setView("3d");
              close();
            }}
            className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[12.5px] font-medium text-white/80 hover:bg-white/10 hover:text-white"
          >
            <Rotate3d className="size-4" /> 3D Studio
          </button>
        )}
        <a
          href={item.url}
          download={item.filename}
          aria-label="Download"
          title="Download"
          className="grid size-8 place-items-center rounded-lg text-white/80 hover:bg-white/10 hover:text-white"
        >
          <Download className="size-4" />
        </a>
        <IconButton label="Close (Esc)" onClick={close} className="text-white/80 hover:bg-white/10 hover:text-white">
          <X />
        </IconButton>
      </div>
      <div
        className="flex min-h-0 flex-1 items-center justify-center p-4 pt-0"
        onClick={(e) => e.target === e.currentTarget && close()}
      >
        {video ? (
          <video src={item.url} controls autoPlay loop playsInline className="max-h-full max-w-full rounded-lg" />
        ) : (
          <img src={item.url} alt={label} className="max-h-full max-w-full rounded-lg object-contain" />
        )}
      </div>
    </div>
  );
}

export function Toasts() {
  const { toasts } = useStudio();
  return (
    <div aria-live="polite" className="pointer-events-none fixed inset-x-0 bottom-5 z-[60] flex flex-col items-center gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="pointer-events-auto flex items-center gap-2 rounded-lg border border-line bg-panel px-3.5 py-2.5 text-[13px] font-medium shadow-float animate-fade-up"
        >
          {t.tone === "error" ? (
            <AlertTriangle className="size-4 text-danger" />
          ) : (
            <CheckCircle2 className="size-4 text-success" />
          )}
          {t.text}
        </div>
      ))}
    </div>
  );
}

export function Scrim() {
  const s = useStudio();
  return (
    <div
      aria-hidden
      onClick={() => s.setSidebarOpen(false)}
      className={clsx(
        "fixed inset-0 z-30 bg-black/50 backdrop-blur-[2px] transition-opacity lg:hidden",
        s.sidebarOpen ? "opacity-100" : "pointer-events-none opacity-0"
      )}
    />
  );
}
