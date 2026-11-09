/* localStorage helpers. These keys are stable: changing one forgets the user's
 * session, transcript or settings. */

import { newId } from "./media";

export const KEYS = {
  session: "imagebot.session",
  history: "imagebot.history",
  theme: "imagebot.theme",
  settings: "imagebot.settings",
} as const;

export function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : ((JSON.parse(raw) as T) ?? fallback);
  } catch {
    return fallback;
  }
}

export function writeJSON(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or quota: the app still works, it just won't remember */
  }
}

export function getSessionId(): string {
  try {
    let id = localStorage.getItem(KEYS.session);
    if (!id) {
      id = newId();
      localStorage.setItem(KEYS.session, id);
    }
    return id;
  } catch {
    return newId();
  }
}
