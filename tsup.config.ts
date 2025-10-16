import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/cli.ts'],
  format: ['esm', 'cjs'],
  target: 'node18',
  sourcemap: true,
  dts: false,
  clean: true,
  minify: false,
  splitting: false,
  shims: false
});
