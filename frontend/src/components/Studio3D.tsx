import clsx from "clsx";
import { AlertTriangle, Box, Clapperboard, Download, ImageUp, Info, Rotate3d, Upload, X } from "lucide-react";
import { createElement, useEffect, useRef, useState, type ReactNode } from "react";
import type { Job3D, Tool3D } from "../lib/types";
import { ANGLES, MOVES, useScene3D } from "../scene3d";
import { useStudio } from "../store";
import { Button, Field, IconButton, Segmented, Spinner, buttonClass } from "./ui";

const TOOL_INFO: Record<Tool3D, { cost: string; time: string }> = {
  view: { cost: "Nano Banana 2", time: "about 10 s" },
  orbit: { cost: "Veo 3.1 Fast", time: "about 1–2 min" },
  mesh: { cost: "Hunyuan3D-2.1 · free", time: "about 2 min" },
};

export function Studio3D() {
  const d = useScene3D();
  return d.source ? <Workspace /> : <UploadScreen />;
}

/* ───────────────────────── empty state: pick a photo ───────────────────────── */

function UploadScreen() {
  const d = useScene3D();
  const s = useStudio();
  const latestImage = s.gallery.find((g) => g.type === "image" && !/-(view|3d)-/.test(g.filename));

  return (
    <main className="scroll-thin flex min-h-0 flex-1 overflow-y-auto bg-bg">
      <div className="m-auto w-full max-w-xl p-4 py-12 sm:p-6">
        <h1 className="text-center text-[26px] font-semibold tracking-tight">See your photo from any angle</h1>
        <p className="mx-auto mt-2 max-w-md text-center text-[14px] text-muted">
          Upload a photo to render it from new viewpoints, fly the camera around it, or build a 3D model you can
          rotate.
        </p>

        <DropZone className="mt-8" />

        {latestImage && (
          <button
            type="button"
            onClick={() => d.setSource({ url: latestImage.url, filename: latestImage.filename })}
            className="mt-3 flex w-full items-center gap-3 rounded-lg border border-line bg-panel p-2 text-left transition-colors hover:bg-subtle"
          >
            <img src={latestImage.url} alt="" className="size-10 rounded object-cover" />
            <span className="min-w-0 flex-1">
              <span className="block text-[13px] font-medium">Use your latest generation</span>
              <span className="block truncate text-[11.5px] text-faint">{latestImage.filename}</span>
            </span>
          </button>
        )}

        <Caveat className="mt-6" />
      </div>
    </main>
  );
}

function DropZone({ className, compact }: { className?: string; compact?: boolean }) {
  const d = useScene3D();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const pick = (files: FileList | null) => {
    const file = files?.[0];
    if (file) d.upload(file);
  };

  return (
    <div className={className}>
      <button
        type="button"
        disabled={d.uploading}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          pick(e.dataTransfer.files);
        }}
        className={clsx(
          "flex w-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed text-center transition-colors",
          compact ? "p-4" : "p-10",
          over ? "border-fg bg-subtle" : "border-line-strong bg-panel hover:bg-subtle"
        )}
      >
        {d.uploading ? <Spinner className="size-5 text-muted" /> : <ImageUp className="size-6 text-muted" />}
        <span className="text-[13.5px] font-medium">{d.uploading ? "Uploading…" : "Upload a photo"}</span>
        {!compact && <span className="text-[12px] text-faint">Drag and drop, or click to browse · JPG, PNG, WebP</span>}
      </button>
      <input
        ref={input}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
      {d.uploadError && <p className="mt-2 text-[12px] text-danger">{d.uploadError}</p>}
    </div>
  );
}

function Caveat({ className }: { className?: string }) {
  return (
    <p className={clsx("flex gap-2 text-[11.5px] leading-snug text-faint", className)}>
      <Info className="mt-px size-3.5 shrink-0" />
      <span>
        A photo only holds what the camera saw. Anything hidden — like a face turned away — is imagined by the AI from
        what is visible, so it can look convincing but may not match the real person.
      </span>
    </p>
  );
}

/* ───────────────────────── workspace: controls + viewer ───────────────────────── */

function Workspace() {
  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto lg:grid-cols-[320px_minmax(0,1fr)] lg:overflow-hidden">
      <Controls />
      <Viewer />
    </div>
  );
}

function Controls() {
  const d = useScene3D();
  const s = useStudio();
  const [tool, setTool] = useState<Tool3D>("view");
  const [angle, setAngle] = useState<string>("left");
  const [move, setMove] = useState<string>("orbit-left");
  const [focus, setFocus] = useState("");
  const google = (s.cfg?.google_models.length ?? 0) > 0;
  const needsGoogle = tool !== "mesh" && s.cfg !== null && !google;
  const pending = d.jobs.some((j) => j.tool === tool && j.status === "pending");

  const go = () => {
    if (tool === "view") d.run("view", ANGLES.find((a) => a.id === angle)!.label, { angle, focus });
    else if (tool === "orbit") d.run("orbit", MOVES.find((m) => m.id === move)!.label, { move, focus });
    else d.run("mesh", "3D model", {});
  };

  return (
    <aside className="flex min-h-0 flex-col border-b border-line bg-panel lg:border-r lg:border-b-0">
      <div className="scroll-thin min-h-0 flex-1 space-y-6 p-4 lg:overflow-y-auto">
        <div className="flex items-center gap-3">
          <img src={d.source!.url} alt="Source" className="size-14 rounded-md border border-line object-cover" />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-medium">Source photo</div>
            <div className="truncate text-[11.5px] text-faint">{d.source!.filename}</div>
          </div>
          <IconButton label="Use a different photo" bordered onClick={() => d.setSource(null)}>
            <Upload />
          </IconButton>
        </div>

        {s.cfg && !google && (
          <div className="flex gap-2 rounded-md border border-warning/30 bg-warning/5 p-3 text-[12px] leading-snug">
            <AlertTriangle className="mt-px size-3.5 shrink-0 text-warning" />
            <span>
              <b className="font-medium">Angles and Orbit are off:</b> the server has no Google key. Put{" "}
              <code className="font-mono">key.json</code> back in the project folder and restart the server.
            </span>
          </div>
        )}

        <Segmented<Tool3D>
          label="Tool"
          className="w-full"
          value={tool}
          onChange={setTool}
          options={[
            { value: "view", label: <><Rotate3d /> Angles</>, title: "Render the scene from a new viewpoint" },
            { value: "orbit", label: <><Clapperboard /> Orbit</>, title: "A camera move around the scene" },
            { value: "mesh", label: <><Box /> 3D</>, title: "A 3D model you can rotate" },
          ]}
        />

        {tool === "view" && (
          <Field label="Camera angle">
            <ChoiceGrid items={ANGLES} value={angle} onChange={setAngle} />
          </Field>
        )}

        {tool === "orbit" && (
          <Field label="Camera move">
            <ChoiceGrid items={MOVES} value={move} onChange={setMove} />
          </Field>
        )}

        {tool !== "mesh" ? (
          <Field
            label="Focus on (optional)"
            htmlFor="focus"
            note="Name the person whose face you want to see, by position or clothing."
          >
            <input
              id="focus"
              value={focus}
              onChange={(e) => setFocus(e.target.value)}
              placeholder="e.g. the woman on the right in the red jacket"
              className="h-9 w-full rounded-md border border-line bg-panel px-3 text-[13px] text-fg outline-none placeholder:text-faint hover:border-line-strong focus:border-accent"
            />
          </Field>
        ) : (
          <p className="text-[12.5px] leading-relaxed text-muted">
            Builds a textured 3D mesh you can drag to rotate. Works best with one person or object on a plain
            background; group photos tend to merge into one shape — use <b className="font-medium text-fg">Angles</b>{" "}
            for those.
          </p>
        )}

        <div className="space-y-2">
          <Button variant="primary" className="w-full" onClick={go} disabled={needsGoogle}>
            {pending && <Spinner className="size-3.5" />}
            {tool === "view" ? "Render this angle" : tool === "orbit" ? "Render orbit clip" : "Build 3D model"}
          </Button>
          <p className="text-center text-[11.5px] text-faint">
            {needsGoogle
              ? "Needs the Google key (key.json) on the server."
              : `${TOOL_INFO[tool].cost} · ${TOOL_INFO[tool].time}`}
          </p>
        </div>

        <Caveat />
      </div>
    </aside>
  );
}

function ChoiceGrid({
  items,
  value,
  onChange,
}: {
  items: readonly { id: string; label: string }[];
  value: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-1.5" role="radiogroup">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="radio"
          aria-checked={value === item.id}
          onClick={() => onChange(item.id)}
          className={clsx(
            "h-8 rounded-md border px-2 text-[12.5px] font-medium transition-colors",
            value === item.id ? "border-fg bg-fg text-bg" : "border-line text-muted hover:border-line-strong hover:text-fg"
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

/* ───────────────────────── the viewer ───────────────────────── */

function Viewer() {
  const d = useScene3D();
  const job = d.jobs.find((j) => j.id === d.selected) ?? null;
  const original = d.source!;

  return (
    <main className="flex min-h-0 min-w-0 flex-col bg-bg">
      <div className="flex min-h-0 flex-1 flex-col gap-4 p-4 sm:p-6 lg:overflow-y-auto">
        <div className="canvas-grid relative flex min-h-[360px] flex-1 items-center justify-center overflow-hidden rounded-lg border border-line">
          {!job ? (
            <img src={original.url} alt="Original" className="max-h-[min(64vh,760px)] max-w-full rounded-md object-contain p-4" />
          ) : job.status === "pending" ? (
            <PendingView job={job} />
          ) : job.status === "error" ? (
            <ErrorView job={job} />
          ) : job.tool === "mesh" ? (
            <ModelViewer src={job.url!} />
          ) : job.tool === "orbit" ? (
            <video
              key={job.url}
              src={job.url}
              controls
              autoPlay
              loop
              playsInline
              className="max-h-[min(64vh,760px)] max-w-full rounded-md p-4"
            />
          ) : (
            <img src={job.url} alt={job.label} className="max-h-[min(64vh,760px)] max-w-full rounded-md object-contain p-4" />
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-[13.5px] font-medium">{job ? job.label : "Original photo"}</div>
            <div className="text-[11.5px] text-faint">
              {!job
                ? "Pick a tool on the left to render new views."
                : job.tool === "mesh" && job.status === "done"
                  ? "Drag to rotate · scroll to zoom · right-drag to pan"
                  : job.tool === "view"
                    ? "New angle · AI-rendered"
                    : job.tool === "orbit"
                      ? "Orbit clip · AI-rendered"
                      : "3D model"}
            </div>
          </div>
          {job?.status === "done" && (
            <a href={job.url} download={job.filename} className={buttonClass("secondary", "sm")}>
              <Download /> {job.tool === "mesh" ? "Download .glb" : "Download"}
            </a>
          )}
        </div>

        <Strip />
      </div>
    </main>
  );
}

function Strip() {
  const d = useScene3D();
  return (
    <div className="scroll-thin flex gap-2 overflow-x-auto pb-1">
      <Tile active={d.selected === null} onClick={() => d.setSelected(null)} label="Original">
        <img src={d.source!.url} alt="" className="size-full object-cover" />
      </Tile>
      {[...d.jobs].reverse().map((job) => (
        <Tile
          key={job.id}
          active={d.selected === job.id}
          onClick={() => d.setSelected(job.id)}
          onRemove={job.status !== "pending" ? () => d.remove(job.id) : undefined}
          label={job.label}
        >
          {job.status === "pending" ? (
            <span className="grid size-full place-items-center bg-panel">
              <Spinner className="size-4 text-faint" />
            </span>
          ) : job.status === "error" ? (
            <span className="grid size-full place-items-center bg-danger/10 text-danger">
              <AlertTriangle className="size-4" />
            </span>
          ) : job.tool === "view" ? (
            <img src={job.url} alt="" className="size-full object-cover" />
          ) : job.tool === "orbit" ? (
            <video src={`${job.url}#t=0.5`} muted preload="metadata" className="size-full object-cover" />
          ) : (
            <span className="grid size-full place-items-center bg-panel text-muted">
              <Box className="size-5" />
            </span>
          )}
        </Tile>
      ))}
    </div>
  );
}

function Tile({
  active,
  onClick,
  onRemove,
  label,
  children,
}: {
  active: boolean;
  onClick: () => void;
  onRemove?: () => void;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="group relative w-20 shrink-0">
      <button
        type="button"
        onClick={onClick}
        title={label}
        className={clsx(
          "block size-20 overflow-hidden rounded-md border bg-raised transition-opacity",
          active ? "border-fg ring-1 ring-fg" : "border-line opacity-75 hover:opacity-100"
        )}
      >
        {children}
      </button>
      <div className="mt-1 truncate text-center text-[11px] text-faint">{label}</div>
      {onRemove && (
        <button
          type="button"
          aria-label={`Remove ${label}`}
          onClick={onRemove}
          className="absolute -top-1.5 -right-1.5 hidden size-5 place-items-center rounded-full border border-line bg-panel text-muted group-hover:grid hover:text-fg"
        >
          <X className="size-3" />
        </button>
      )}
    </div>
  );
}

function PendingView({ job }: { job: Job3D }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="flex flex-col items-center gap-2.5 text-center">
      <Spinner className="size-5 text-muted" />
      <div className="text-[13px] font-medium">Rendering {job.label.toLowerCase()}…</div>
      <div className="font-mono text-[11.5px] text-faint tabular-nums">
        {((now - job.startedAt) / 1000).toFixed(0)}s · {TOOL_INFO[job.tool].time}
      </div>
    </div>
  );
}

function ErrorView({ job }: { job: Job3D }) {
  return (
    <div className="max-w-lg p-6 text-center">
      <AlertTriangle className="mx-auto size-5 text-danger" />
      <div className="mt-2 text-[13px] font-medium">{job.label} failed</div>
      <code className="mt-2 block font-mono text-xs break-words whitespace-pre-wrap text-muted">{job.error}</code>
    </div>
  );
}

/** <model-viewer> is a web component; load it only when a model is shown. */
function ModelViewer({ src }: { src: string }) {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let alive = true;
    import("@google/model-viewer").then(() => alive && setReady(true));
    return () => {
      alive = false;
    };
  }, []);
  if (!ready) return <Spinner className="size-5 text-muted" />;
  return createElement("model-viewer", {
    src,
    "camera-controls": "",
    "auto-rotate": "",
    "auto-rotate-delay": "1500",
    "shadow-intensity": "1",
    exposure: "1",
    "interaction-prompt": "none",
    style: { position: "absolute", inset: 0, width: "100%", height: "100%", background: "transparent" },
  });
}
