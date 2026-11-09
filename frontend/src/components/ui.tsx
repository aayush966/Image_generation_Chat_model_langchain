import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { forwardRef } from "react";

/* ───────────────────────── Button ───────────────────────── */

type Variant = "primary" | "secondary" | "ghost";
type Size = "sm" | "md";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-fg text-bg hover:opacity-90 active:opacity-80",
  secondary: "border border-line bg-panel text-fg hover:bg-subtle hover:border-line-strong",
  ghost: "text-muted hover:bg-raised hover:text-fg",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-[12.5px] gap-1.5",
  md: "h-9 px-3.5 text-[13px] gap-2",
};

export const buttonClass = (variant: Variant = "secondary", size: Size = "md", extra?: string) =>
  clsx(
    "inline-flex items-center justify-center rounded-md font-medium whitespace-nowrap transition-[background,border,color,opacity] duration-150",
    "disabled:pointer-events-none disabled:opacity-40 [&_svg]:size-4 [&_svg]:shrink-0",
    VARIANTS[variant],
    SIZES[size],
    extra
  );

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", className, type = "button", ...rest },
  ref
) {
  return <button ref={ref} type={type} className={buttonClass(variant, size, className)} {...rest} />;
});

export function IconButton({
  label,
  className,
  children,
  bordered,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; bordered?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={clsx(
        "inline-grid size-8 shrink-0 place-items-center rounded-md text-muted transition-colors hover:bg-raised hover:text-fg [&_svg]:size-4",
        bordered && "border border-line bg-panel",
        className
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/* ───────────────────────── Segmented control ───────────────────────── */

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  title?: string;
  disabled?: boolean;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = "md",
  className,
  disabled,
}: {
  value: T;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
  label: string;
  size?: "sm" | "md";
  className?: string;
  disabled?: boolean;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={clsx("inline-flex rounded-lg border border-line bg-subtle p-0.5", className)}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={o.title}
            disabled={disabled || o.disabled}
            onClick={() => onChange(o.value)}
            className={clsx(
              "inline-flex flex-1 items-center justify-center gap-1.5 rounded-md font-medium transition-colors [&_svg]:size-3.5",
              size === "sm" ? "h-7 px-2 text-xs" : "h-8 px-3 text-[13px]",
              active ? "bg-panel text-fg shadow-sm ring-1 ring-line" : "text-muted hover:text-fg",
              "disabled:cursor-not-allowed disabled:opacity-40"
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/* ───────────────────────── Form fields ───────────────────────── */

export function Field({
  label,
  value,
  note,
  children,
  muted,
  htmlFor,
}: {
  label: string;
  value?: ReactNode;
  note?: ReactNode;
  children: ReactNode;
  muted?: boolean;
  htmlFor?: string;
}) {
  return (
    <div className={clsx("space-y-2 transition-opacity", muted && "opacity-55")}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="text-xs font-medium text-muted">
          {label}
        </label>
        {value !== undefined && <span className="font-mono text-[11.5px] text-fg tabular-nums">{value}</span>}
      </div>
      {children}
      {note && <p className="text-[11.5px] leading-snug text-faint">{note}</p>}
    </div>
  );
}

export function Slider({
  id,
  value,
  min,
  max,
  step = 1,
  onChange,
  disabled,
}: {
  id?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <input
      id={id}
      type="range"
      className="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      style={{ ["--fill" as string]: `${fill}%` }}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Switch({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  description?: string;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="min-w-0">
        <span className="block text-xs font-medium text-muted">{label}</span>
        {description && <span className="mt-0.5 block text-[11.5px] leading-snug text-faint">{description}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={clsx(
          "relative mt-0.5 inline-flex h-5 w-9 shrink-0 rounded-full transition-colors",
          checked ? "bg-fg" : "bg-line-strong"
        )}
      >
        <span
          className={clsx(
            "absolute top-0.5 size-4 rounded-full bg-panel shadow transition-transform",
            checked ? "translate-x-[18px]" : "translate-x-0.5"
          )}
        />
      </button>
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={clsx("inline-block size-4 animate-spin rounded-full border-2 border-current border-r-transparent", className)}
    />
  );
}
