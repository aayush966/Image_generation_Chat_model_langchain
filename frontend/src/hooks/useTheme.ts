import { useCallback, useSyncExternalStore } from "react";
import { KEYS } from "../lib/storage";
import type { Theme } from "../lib/types";

/* The theme lives on <html data-theme>, set before first paint by index.html.
 * Every component reading it subscribes to that attribute. */

const listeners = new Set<() => void>();
const read = (): Theme => (document.documentElement.dataset.theme === "light" ? "light" : "dark");

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, read);
  const toggle = useCallback(() => {
    const next: Theme = read() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEYS.theme, next);
    } catch {
      /* not persisted, still applied */
    }
    listeners.forEach((l) => l());
  }, []);
  return { theme, toggle };
}
