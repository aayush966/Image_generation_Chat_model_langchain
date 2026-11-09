import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The FastAPI backend (server.py) runs on :8000. In dev, Vite proxies the API and
// the generated media to it; in production FastAPI serves the built `dist/`.
const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": BACKEND,
      "/media": BACKEND,
      "/images": BACKEND,
    },
  },
});
