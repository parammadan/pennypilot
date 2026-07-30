import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// The storefront must build to ONE self-contained HTML file: the demo
// driver writes it to a temp dir with PRODUCTS injected and loads it via
// file:// — no server, no asset requests.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
});
