import clsx from "clsx";
import { Film } from "lucide-react";
import { useRef } from "react";
import { isVideo } from "../lib/media";
import type { MediaItem } from "../lib/types";

/** An <img>, or a muted clip that plays only on hover so a grid of videos
 * does not fight for bandwidth. */
export function Thumb({ item, alt, className }: { item: MediaItem; alt: string; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const cls = clsx("size-full object-cover", className);

  if (isVideo(item)) {
    return (
      <video
        ref={ref}
        // The time fragment makes browsers paint the first frame instead of black.
        src={`${item.url}#t=0.1`}
        muted
        loop
        playsInline
        preload="metadata"
        aria-label={alt}
        className={cls}
        onMouseEnter={() => ref.current?.play().catch(() => {})}
        onMouseLeave={() => {
          if (!ref.current) return;
          ref.current.pause();
          ref.current.currentTime = 0;
        }}
      />
    );
  }
  return <img src={item.url} alt={alt} loading="lazy" decoding="async" className={cls} />;
}

export function VideoBadge() {
  return (
    <span className="pointer-events-none absolute top-2 left-2 inline-flex items-center gap-1 rounded-md bg-black/60 px-1.5 py-0.5 text-[10.5px] font-medium text-white backdrop-blur">
      <Film className="size-3" /> Video
    </span>
  );
}
