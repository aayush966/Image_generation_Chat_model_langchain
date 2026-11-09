import { useEffect } from "react";
import { Gallery } from "./components/Gallery";
import { Header } from "./components/Header";
import { Lightbox, Scrim, Toasts } from "./components/Overlays";
import { Sidebar } from "./components/Sidebar";
import { Stage } from "./components/Stage";
import { Studio3D } from "./components/Studio3D";
import { Scene3DProvider } from "./scene3d";
import { StudioProvider, useStudio } from "./store";

export default function App() {
  return (
    <StudioProvider>
      <Scene3DProvider>
        <Shell />
      </Scene3DProvider>
    </StudioProvider>
  );
}

function Shell() {
  const s = useStudio();
  useShortcuts();

  return (
    <div className="flex h-full flex-col">
      <Header />
      {s.view === "3d" ? (
        <Studio3D />
      ) : (
        /* settings | workspace */
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[288px_minmax(0,1fr)]">
          <Sidebar />
          {s.view === "gallery" ? <Gallery /> : <Stage />}
        </div>
      )}
      <Scrim />
      <Lightbox />
      <Toasts />
    </div>
  );
}

/** Ctrl+M flips image/video · Esc closes overlays. */
function useShortcuts() {
  const s = useStudio();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (s.lightbox) s.setLightbox(null);
        else s.setSidebarOpen(false);
        return;
      }
      if (!(e.ctrlKey || e.metaKey) || e.shiftKey || e.altKey) return;
      if (e.key.toLowerCase() === "m") {
        e.preventDefault();
        if (!s.busy) s.setMode(s.mode === "video" ? "image" : "video");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [s]);
}
