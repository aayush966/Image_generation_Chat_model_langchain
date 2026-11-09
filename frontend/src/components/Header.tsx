import clsx from "clsx";
import { Moon, Plus, SlidersHorizontal, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { useTheme } from "../hooks/useTheme";
import { BACKEND_STATUS, useStudio } from "../store";
import { Button, IconButton } from "./ui";

export function Header() {
  const s = useStudio();
  const { theme, toggle } = useTheme();

  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line bg-panel px-3 sm:px-4">
      {s.view !== "3d" && (
        <IconButton
          label="Settings"
          className="lg:hidden"
          aria-expanded={s.sidebarOpen}
          onClick={() => s.setSidebarOpen(true)}
        >
          <SlidersHorizontal />
        </IconButton>
      )}

      <div className="flex items-center gap-2 pr-2 lg:w-[272px]">
        <Logo />
        <span className="hidden text-[14px] font-semibold tracking-tight sm:inline">ImageBot</span>
      </div>

      <nav className="flex items-center gap-1" aria-label="Main">
        <Tab
          active={s.view === "create"}
          onClick={() => s.setView("create")}
        >
          Create
        </Tab>
        <Tab
          active={s.view === "gallery"}
          onClick={() => {
            s.setView("gallery");
            s.loadGallery();
          }}
        >
          Gallery
          {s.gallery.length > 0 && <span className="ml-1.5 text-faint tabular-nums">{s.gallery.length}</span>}
        </Tab>
        <Tab active={s.view === "3d"} onClick={() => s.setView("3d")}>
          3D Studio
        </Tab>
      </nav>

      <div className="ml-auto flex items-center gap-1.5">
        <Status />
        <Button
          size="sm"
          className={s.view === "3d" ? "!hidden" : undefined}
          onClick={s.newChat}
          title={`Clear the conversation (${s.remembered} message${s.remembered === 1 ? "" : "s"} in memory)`}
        >
          <Plus /> <span className="hidden sm:inline">New chat</span>
        </Button>
        <IconButton label={theme === "dark" ? "Light theme" : "Dark theme"} onClick={toggle}>
          {theme === "dark" ? <Sun /> : <Moon />}
        </IconButton>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <span className="grid size-7 place-items-center rounded-md bg-fg text-bg" aria-hidden>
      <svg viewBox="0 0 16 16" className="size-3.5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <rect x="2" y="2" width="12" height="12" rx="2.5" />
        <path d="M2.5 11.5 6 8l2.5 2.5L10.5 8.5l3 3" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx="10.5" cy="5.5" r="1" fill="currentColor" stroke="none" />
      </svg>
    </span>
  );
}

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={clsx(
        "inline-flex h-8 items-center rounded-md px-3 text-[13px] font-medium transition-colors",
        active ? "bg-raised text-fg" : "text-muted hover:text-fg"
      )}
    >
      {children}
    </button>
  );
}

function Status() {
  const s = useStudio();
  const backend = s.cfg ? BACKEND_STATUS[s.cfg.backend] ?? { tone: "warn" as const, label: s.cfg.backend } : null;
  const tone = s.status === "offline" ? "error" : s.status === "loading" ? "warn" : backend!.tone;
  const label =
    s.status === "offline" ? "Offline" : s.status === "loading" ? "Connecting" : backend!.label;
  const title =
    s.status === "offline"
      ? "Can't reach the server. Start server.py on port 8000."
      : s.cfg
        ? `Backend: ${s.cfg.backend} · ${s.cfg.has_token ? "token configured" : "no token"}`
        : undefined;

  return (
    <span
      title={title}
      className="mr-1 hidden h-8 items-center gap-2 rounded-md px-2 text-[12px] text-muted md:inline-flex"
    >
      <span
        className={clsx(
          "size-1.5 rounded-full",
          tone === "ok" ? "bg-success" : tone === "warn" ? "bg-warning" : "bg-danger"
        )}
      />
      {label}
    </span>
  );
}
