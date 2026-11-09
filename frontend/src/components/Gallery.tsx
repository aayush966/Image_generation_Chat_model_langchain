import { ImageOff, RefreshCw } from "lucide-react";
import { useState } from "react";
import { isVideo, labelOf, timeAgo } from "../lib/media";
import { useStudio } from "../store";
import { Thumb, VideoBadge } from "./Thumb";
import { Button, IconButton, Segmented } from "./ui";

type Filter = "all" | "image" | "video";

export function Gallery() {
  const s = useStudio();
  const [filter, setFilter] = useState<Filter>("all");
  const videos = s.gallery.filter(isVideo).length;
  const images = s.gallery.length - videos;
  const items = s.gallery.filter((item) => filter === "all" || (filter === "video") === isVideo(item));

  return (
    <section className="flex min-h-0 min-w-0 flex-col bg-bg">
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-line bg-panel px-4 py-3 sm:px-6">
        <div className="mr-auto">
          <h2 className="text-[15px] font-semibold tracking-tight">Gallery</h2>
          <p className="text-[12px] text-faint">
            {s.gallery.length
              ? `${images} image${images === 1 ? "" : "s"} · ${videos} video${videos === 1 ? "" : "s"}`
              : "Nothing generated yet"}
          </p>
        </div>
        <Segmented<Filter>
          label="Filter"
          size="sm"
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "All" },
            { value: "image", label: "Images" },
            { value: "video", label: "Videos" },
          ]}
        />
        <IconButton label="Refresh" bordered onClick={s.loadGallery}>
          <RefreshCw />
        </IconButton>
      </header>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
        {items.length === 0 ? (
          <div className="flex h-full min-h-[320px] flex-col items-center justify-center gap-3 text-center">
            <span className="grid size-11 place-items-center rounded-lg border border-line bg-panel text-faint">
              <ImageOff className="size-5" />
            </span>
            <div className="text-[14px] font-medium">No {filter === "all" ? "generations" : `${filter}s`} yet</div>
            <p className="max-w-xs text-[12.5px] text-faint">Everything you generate is saved here automatically.</p>
            <Button variant="primary" size="sm" onClick={() => s.setView("create")}>
              Start creating
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
            {items.map((item) => {
              const label = labelOf(item);
              return (
                <figure
                  key={item.filename}
                  className="group relative aspect-square cursor-pointer overflow-hidden rounded-lg border border-line bg-raised"
                  onClick={() => s.setLightbox(item)}
                >
                  <Thumb item={item} alt={label} className="transition-transform duration-300 group-hover:scale-[1.03]" />
                  {isVideo(item) && <VideoBadge />}
                  <figcaption className="absolute inset-x-0 bottom-0 translate-y-1 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3 pt-8 opacity-0 transition-all duration-200 group-hover:translate-y-0 group-hover:opacity-100">
                    <div className="line-clamp-2 text-[12.5px] font-medium text-white">{label}</div>
                    {item.created && <div className="mt-0.5 text-[11px] text-white/70">{timeAgo(item.created)}</div>}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
