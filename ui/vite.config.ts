import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/stream':   { target: 'ws://localhost:8080',   ws: true, changeOrigin: true },
      '/replay':   { target: 'ws://localhost:8081',   ws: true, changeOrigin: true },
      '/metrics':  { target: 'http://localhost:9090',           changeOrigin: true },
      '/chapters': { target: 'http://localhost:9090',           changeOrigin: true },
    },
  },
});
