import path from 'node:path';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  root: path.resolve(__dirname, 'app'),
  plugins: [react()],
  resolve: {
    alias: {
      '@core': path.resolve(__dirname, 'src/core'),
      '@src': path.resolve(__dirname, 'src')
    }
  },
  server: {
    fs: {
      allow: [path.resolve(__dirname)]
    }
  },
  build: {
    outDir: path.resolve(__dirname, 'dist-app'),
    emptyOutDir: true
  }
});
