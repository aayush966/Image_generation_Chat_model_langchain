import clsx from "clsx";
import { ArrowUp, Sparkles } from "lucide-react";
import { useEffect, useLayoutEffect, useRef } from "react";
import { MODE_COPY, useStudio } from "../store";
import { Spinner } from "./ui";

export function Composer() {
  const s = useStudio();
  const ref = useRef<HTMLTextAreaElement>(null);
  const copy = MODE_COPY[s.mode];
  const video = s.mode === "video";

  // Grow with the prompt, up to a cap.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [s.draft]);

  // Back to the prompt whenever a turn finishes or the mode changes.
  useEffect(() => {
    if (!s.busy) ref.current?.focus();
  }, [s.busy, s.mode]);

  const submit = () => {
    const text = s.draft;
    if (!text.trim() || s.busy) return;
    s.setDraft("");
    s.send(text);
  };

  return (
    <form
      className="shrink-0 border-t border-line bg-panel px-4 py-3 sm:px-6 sm:py-4"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="mx-auto max-w-5xl rounded-lg border border-line bg-bg transition-colors focus-within:border-line-strong">
        <textarea
          ref={ref}
          rows={1}
          value={s.draft}
          onChange={(e) => s.setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={copy.placeholder}
          aria-label="Prompt"
          className="block max-h-[180px] min-h-[52px] w-full resize-none bg-transparent px-3.5 pt-3 pb-1 text-[14px] leading-relaxed text-fg outline-none placeholder:text-faint"
        />
        <div className="flex items-center gap-2 px-2 pb-2">
          <button
            type="button"
            aria-pressed={s.settings.enhance}
            onClick={() => s.updateSettings({ enhance: !s.settings.enhance })}
            title="Expand short prompts with more detail before generating"
            className={clsx(
              "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs font-medium transition-colors [&>svg]:size-3.5",
              s.settings.enhance ? "bg-raised text-fg" : "text-faint hover:text-muted"
            )}
          >
            <Sparkles /> Enhance prompt
          </button>
          <span className="hidden font-mono text-[11.5px] text-faint sm:inline">
            {s.size[0]}×{s.size[1]}
            {video ? ` · ${s.settings.duration.toFixed(1)}s` : ""}
          </span>

          <button
            type="submit"
            disabled={s.busy || !s.draft.trim()}
            className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-md bg-fg px-3.5 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-35"
          >
            {s.busy ? <Spinner className="size-3.5" /> : <ArrowUp className="size-4" />}
            {s.busy ? "Generating" : copy.button}
          </button>
        </div>
      </div>
    </form>
  );
}
