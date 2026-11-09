import clsx from "clsx";
import {
  ChevronDown,
  Dices,
  Image as ImageIcon,
  RectangleHorizontal,
  RectangleVertical,
  Square,
  Video,
  X,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { shortModel } from "../lib/media";
import type { Aspect, Mode } from "../lib/types";
import { modelDefaults, modelOptions, resolutionFor, useStudio } from "../store";
import { Field, IconButton, Segmented, Slider, Switch } from "./ui";

/** Generation settings. A column on desktop, an off-canvas drawer on mobile. */
export function Sidebar() {
  const s = useStudio();
  const { settings, cfg, mode, model } = s;
  const video = mode === "video";
  const options = modelOptions(cfg, mode);
  const defaults = modelDefaults(cfg, mode, model);
  const google = Boolean(model && cfg?.google_models.includes(model));

  const modelNote = google
    ? video
      ? "Veo renders 4, 6 or 8 second clips."
      : undefined
    : defaults
      ? video
        ? `${defaults.fps} fps · tuned for ${defaults.steps} steps`
        : `Tuned for ${defaults.steps} steps`
      : undefined;

  return (
    <aside
      aria-label="Settings"
      className={clsx(
        "flex min-h-0 flex-col border-r border-line bg-panel",
        "fixed inset-y-0 left-0 z-40 w-[300px] transition-transform duration-200 lg:static lg:z-auto lg:w-auto lg:translate-x-0",
        s.sidebarOpen ? "translate-x-0 shadow-float" : "-translate-x-full"
      )}
    >
      <div className="flex h-14 shrink-0 items-center border-b border-line px-4 lg:hidden">
        <span className="text-[14px] font-semibold">Settings</span>
        <IconButton label="Close settings" className="ml-auto" onClick={() => s.setSidebarOpen(false)}>
          <X />
        </IconButton>
      </div>

      <div className="scroll-thin min-h-0 flex-1 space-y-6 overflow-y-auto p-4">
        <Segmented<Mode>
          label="Generation mode"
          className="w-full"
          value={mode}
          disabled={s.busy}
          onChange={s.setMode}
          options={[
            { value: "image", label: <><ImageIcon /> Image</>, title: "Text to image (Ctrl+M)" },
            { value: "video", label: <><Video /> Video</>, title: "Text to video (Ctrl+M)" },
          ]}
        />

        <Field label="Model" htmlFor="model" note={modelNote}>
          <div className="relative">
            <select
              id="model"
              value={model ?? ""}
              onChange={(e) => s.setModel(e.target.value)}
              disabled={!cfg}
              className="h-9 w-full appearance-none rounded-md border border-line bg-panel pr-8 pl-3 text-[13px] font-medium text-fg outline-none transition-colors hover:border-line-strong focus:border-accent"
            >
              {!cfg && <option>Loading models…</option>}
              {options.map((o) => (
                <option key={o.id} value={o.id} disabled={o.disabled}>
                  {o.label}
                  {o.disabled ? " (needs token)" : ""}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute top-1/2 right-2.5 size-4 -translate-y-1/2 text-faint" />
          </div>
        </Field>

        <Field label="Aspect ratio">
          <Segmented<Aspect>
            label="Aspect ratio"
            className="w-full"
            value={settings.aspect}
            onChange={(aspect) => s.updateSettings({ aspect })}
            options={[
              { value: "1:1", label: <><Square /> 1:1</>, title: "Square" },
              { value: "16:9", label: <><RectangleHorizontal /> 16:9</>, title: "Landscape" },
              { value: "9:16", label: <><RectangleVertical /> 9:16</>, title: "Portrait" },
            ]}
          />
        </Field>

        <Field label="Resolution" value={`${s.size[0]} × ${s.size[1]}`}>
          <Segmented
            label="Resolution"
            className="w-full"
            value={String(resolutionFor(settings, mode))}
            onChange={s.setResolution}
            options={[
              { value: "512", label: "512" },
              { value: "768", label: "768" },
              { value: "1024", label: "1024", disabled: video, title: video ? "Too slow for video" : undefined },
            ]}
          />
        </Field>

        {video && (
          <Field label="Duration" value={`${settings.duration.toFixed(1)}s`}>
            <Slider
              value={settings.duration}
              min={cfg?.video.min_duration ?? 1}
              max={cfg?.video.max_duration ?? 10}
              onChange={(duration) => s.updateSettings({ duration })}
            />
          </Field>
        )}

        <Advanced>
          <AdvancedSettings google={google} />
        </Advanced>
      </div>
    </aside>
  );
}

function Advanced({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-line pt-4">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-xs font-medium text-muted hover:text-fg"
      >
        Advanced
        <ChevronDown className={clsx("size-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && <div className="mt-4 space-y-5 animate-fade-up">{children}</div>}
    </div>
  );
}

function AdvancedSettings({ google }: { google: boolean }) {
  const s = useStudio();
  const { settings, cfg, mode, model } = s;
  const defaults = modelDefaults(cfg, mode, model);
  // Few-step turbo models run without classifier-free guidance.
  const noGuidance = Boolean(defaults) && defaults!.guidance_scale <= 1;
  const short = shortModel(model);

  return (
    <>
      <Field label="Steps" value={settings.steps} muted={google} note={google ? "Set by the model." : undefined}>
        <Slider value={settings.steps} min={1} max={50} onChange={(steps) => s.updateSettings({ steps })} />
      </Field>

      <Field
        label="Guidance"
        value={settings.guidance.toFixed(1)}
        muted={noGuidance || google}
        note={google ? "Set by the model." : noGuidance ? `${short} ignores guidance.` : undefined}
      >
        <Slider
          value={settings.guidance}
          min={0}
          max={15}
          step={0.5}
          onChange={(guidance) => s.updateSettings({ guidance })}
        />
      </Field>

      <Field label="Seed" htmlFor="seed">
        <div className="flex gap-2">
          <input
            id="seed"
            type="number"
            inputMode="numeric"
            placeholder="Random"
            value={s.seed}
            onChange={(e) => s.setSeed(e.target.value)}
            className="h-9 min-w-0 flex-1 rounded-md border border-line bg-panel px-3 font-mono text-[12.5px] text-fg outline-none placeholder:font-sans placeholder:text-faint hover:border-line-strong focus:border-accent"
          />
          <IconButton
            label="Random seed"
            bordered
            className="size-9"
            onClick={() => s.setSeed(String(Math.floor(Math.random() * 2 ** 31)))}
          >
            <Dices />
          </IconButton>
        </div>
      </Field>

      <Switch
        checked={settings.forceImage}
        onChange={(forceImage) => s.updateSettings({ forceImage })}
        label="Always generate"
        description="Generate from every message instead of letting the assistant decide."
      />
    </>
  );
}
