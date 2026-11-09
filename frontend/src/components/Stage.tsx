import clsx from "clsx";
import { AlertTriangle, Copy, Download, Info, Maximize2, RefreshCw, Rotate3d, X } from "lucide-react";
import { useEffect, useState } from "react";
import { isVideo, labelOf, sessionMedia, shortModel } from "../lib/media";
import type { MediaItem, Pending, Reply } from "../lib/types";
import { useScene3D } from "../scene3d";
import { MODE_COPY, useStudio } from "../store";
import { Composer } from "./Composer";
import { Thumb } from "./Thumb";
import { Button, IconButton, Spinner, buttonClass } from "./ui";

export function Stage() {
  const s = useStudio();
  const { current, pending } = s;
  const showHero = !current && !pending;

  return (
    <main className="flex min-h-0 min-w-0 flex-col bg-bg">
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex min-h-full w-full max-w-5xl flex-col gap-4 p-4 sm:p-6">
          {showHero ? <Hero /> : <Canvas item={current} pending={pending} />}

          {current && !pending && <ResultBar item={current} />}

          {s.replies.length > 0 && (
            <div className="space-y-2">
              {s.replies.map((r) => (
                <ReplyCard key={r.id} reply={r} onDismiss={() => s.dismissReply(r.id)} />
              ))}
            </div>
          )}

          <Filmstrip />
        </div>
      </div>

      <Composer />
    </main>
  );
}

/* ───────────────────────── the canvas ───────────────────────── */

function Canvas({ item, pending }: { item: MediaItem | null; pending: Pending | null }) {
  const s = useStudio();
  const [loaded, setLoaded] = useState(false);
  useEffect(() => setLoaded(false), [item?.url]);

  const video = isVideo(item);
  // While rendering, the frame already takes the shape of the result.
  const w = pending?.width ?? item?.width ?? 1;
  const h = pending?.height ?? item?.height ?? 1;

  return (
    <div className="canvas-grid relative flex min-h-[320px] flex-1 items-center justify-center overflow-hidden rounded-lg border border-line p-4 sm:p-6">
      {pending ? (
        <PendingFrame pending={pending} />
      ) : item ? (
        <div className="group relative max-h-full max-w-full" style={{ aspectRatio: `${w} / ${h}` }}>
          {video ? (
            <video
              key={item.url}
              src={item.url}
              controls
              autoPlay
              loop
              muted
              playsInline
              onLoadedData={() => setLoaded(true)}
              className={clsx(
                "max-h-[min(60vh,720px)] max-w-full rounded-md bg-black shadow-float transition-opacity duration-300",
                loaded ? "opacity-100" : "opacity-0"
              )}
            />
          ) : (
            <img
              key={item.url}
              src={item.url}
              alt={item.prompt || ""}
              onLoad={() => setLoaded(true)}
              onClick={() => s.setLightbox(item)}
              className={clsx(
                "max-h-[min(60vh,720px)] max-w-full cursor-zoom-in rounded-md shadow-float transition-opacity duration-300",
                loaded ? "opacity-100" : "opacity-0"
              )}
            />
          )}
          <IconButton
            label="View full size"
            onClick={() => s.setLightbox(item)}
            className="absolute top-2 right-2 bg-black/55 text-white opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 hover:bg-black/70 hover:text-white"
          >
            <Maximize2 />
          </IconButton>
        </div>
      ) : null}
    </div>
  );
}

function PendingFrame({ pending }: { pending: Pending }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((performance.now() - pending.startedAt) / 1000), 100);
    return () => clearInterval(t);
  }, [pending.startedAt]);

  const landscape = pending.width >= pending.height;
  return (
    <div
      className="shimmer relative flex items-center justify-center overflow-hidden rounded-md border border-line bg-panel"
      style={{
        aspectRatio: `${pending.width} / ${pending.height}`,
        [landscape ? "width" : "height"]: landscape ? "min(100%, 560px)" : "min(56vh, 520px)",
      }}
    >
      <div className="flex flex-col items-center gap-2.5 px-6 text-center">
        <Spinner className="size-5 text-muted" />
        <div className="text-[13px] font-medium">{pending.label}…</div>
        <div className="font-mono text-[11.5px] text-faint tabular-nums">{elapsed.toFixed(1)}s</div>
      </div>
    </div>
  );
}

function Hero() {
  const s = useStudio();
  const copy = MODE_COPY[s.mode];
  return (
    <div className="flex flex-1 flex-col items-center justify-center py-12 text-center animate-fade-up">
      <h1 className="text-[26px] font-semibold tracking-tight">{copy.title}</h1>
      <p className="mt-2 max-w-md text-[14px] text-muted">{copy.text}</p>
      <div className="mt-8 w-full max-w-xl text-left">
        <div className="mb-2 text-[11.5px] font-medium text-faint">Try an example</div>
        <div className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-panel">
          {copy.prompts.map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => s.send(prompt)}
              disabled={s.busy}
              className="block w-full px-4 py-3 text-left text-[13px] text-muted transition-colors hover:bg-subtle hover:text-fg"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ───────────────────────── result details + actions ───────────────────────── */

function ResultBar({ item }: { item: MediaItem }) {
  const s = useStudio();
  const d = useScene3D();
  const [open, setOpen] = useState(false);
  const video = isVideo(item);

  const meta = [
    shortModel(item.model),
    item.width ? `${item.width}×${item.height}` : "",
    video && item.duration ? `${item.duration}s` : "",
    !video && item.steps ? `${item.steps} steps` : "",
    item.seed !== null && item.seed !== undefined ? `seed ${item.seed}` : "",
    item.seconds ? `${item.seconds}s` : "",
  ].filter(Boolean);

  return (
    <section className="flex flex-col gap-3 animate-fade-up md:flex-row md:items-start md:gap-6">
      <div className="min-w-0 flex-1">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          title={open ? "Collapse" : "Expand"}
          className={clsx("block text-left text-[13.5px] leading-relaxed text-fg", !open && "line-clamp-2")}
        >
          {labelOf(item)}
        </button>
        <div className="mt-1 font-mono text-[11.5px] text-faint">{meta.join("  ·  ")}</div>
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        <Button size="sm" onClick={s.vary} disabled={s.busy || !item.prompt} title="Render again with a new seed">
          <RefreshCw /> Vary
        </Button>
        <Button size="sm" onClick={s.copyPrompt} disabled={!item.prompt} title="Copy prompt">
          <Copy /> Copy
        </Button>
        {!video && (
          <Button
            size="sm"
            title="See this image from other angles"
            onClick={() => {
              d.setSource({ url: item.url, filename: item.filename, width: item.width, height: item.height });
              s.setView("3d");
            }}
          >
            <Rotate3d /> 3D
          </Button>
        )}
        <a
          href={item.url}
          download={item.filename || (video ? "clip.mp4" : "image.png")}
          className={buttonClass("primary", "sm")}
        >
          <Download /> Download
        </a>
      </div>
    </section>
  );
}

/** This session's results, newest first — click to bring one back to the canvas. */
function Filmstrip() {
  const s = useStudio();
  const items = [...sessionMedia(s.history)].reverse();
  if (items.length < 2) return null;

  return (
    <section className="border-t border-line pt-4">
      <div className="mb-2 text-[11.5px] font-medium text-faint">This session</div>
      <div className="scroll-thin flex gap-2 overflow-x-auto pb-1">
        {items.map((item) => {
          const active = s.current?.filename === item.filename;
          return (
            <button
              key={item.filename}
              type="button"
              title={labelOf(item)}
              onClick={() => s.showItem({ ...item, prompt: labelOf(item) })}
              className={clsx(
                "relative size-16 shrink-0 overflow-hidden rounded-md border bg-raised transition-opacity",
                active ? "border-fg ring-1 ring-fg" : "border-line opacity-70 hover:opacity-100"
              )}
            >
              <Thumb item={item} alt={labelOf(item)} />
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ReplyCard({ reply, onDismiss }: { reply: Reply; onDismiss: () => void }) {
  const fail = reply.kind === "fail";
  const notice = reply.kind === "notice";

  return (
    <div
      className={clsx(
        "flex gap-3 rounded-lg border p-3.5 animate-fade-up",
        fail ? "border-danger/30 bg-danger/5" : notice ? "border-warning/30 bg-warning/5" : "border-line bg-panel"
      )}
    >
      {(fail || notice) && (
        <span className={clsx("mt-0.5 shrink-0 [&>svg]:size-4", fail ? "text-danger" : "text-warning")}>
          {fail ? <AlertTriangle /> : <Info />}
        </span>
      )}
      <div className="min-w-0 flex-1 text-[13px] leading-relaxed">
        {fail && <div className="mb-1 font-medium">Generation failed</div>}
        {fail ? (
          <code className="block font-mono text-xs break-words whitespace-pre-wrap text-muted">{reply.text}</code>
        ) : (
          <Linkified text={reply.text} />
        )}
        {fail && reply.retry && (
          <Button
            size="sm"
            className="mt-3"
            onClick={() => {
              onDismiss();
              reply.retry!();
            }}
          >
            <RefreshCw /> Try again
          </Button>
        )}
      </div>
      <IconButton label="Dismiss" className="-mt-1 -mr-1 size-7" onClick={onDismiss}>
        <X />
      </IconButton>
    </div>
  );
}

/** Render the first URL in a message as a link. */
function Linkified({ text }: { text: string }) {
  const match = text.match(/https?:\/\/\S+?(?=[)\s]|$)/);
  if (!match || match.index === undefined) return <p>{text}</p>;
  return (
    <p>
      {text.slice(0, match.index)}
      <a href={match[0]} target="_blank" rel="noreferrer" className="font-medium text-accent underline underline-offset-2">
        {match[0]}
      </a>
      {text.slice(match.index + match[0].length)}
    </p>
  );
}
