import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const base = process.env.VITE_BASE_PATH || "/AlphaKyoen/";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base,
});
