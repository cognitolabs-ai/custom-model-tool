# Operations Guide

This document covers day-to-day tasks for working on the Codex Notebook Generator project.

## Prerequisites

- Node.js 18.x or newer
- pnpm 8+
- Git (for version control)
- Optional: Docker 24+ for containerised workflows

Install dependencies once:

```bash
pnpm install
```

## Common Commands

| Task | Command |
|------|---------|
| Type-check & build core library | `pnpm build` |
| Build the React UI | `pnpm build:ui` |
| Run tests | `pnpm test` |
| Watch `tsup` build | `pnpm dev` |
| Launch the UI dev server | `pnpm dev:ui` |
| Generate artifacts from CLI | `pnpm generate --config <file> --out <dir>` |

## Repository Layout

```
app/            # React SPA (Vite + Tailwind)
dist/           # tsup output (core generator / CLI)
dist-app/       # Vite build output for UI
schemas/        # JSON schema describing generator input
src/            # Core generator (TS)
docs/           # Documentation (architecture, configuration, operations)
examples/       # Sample configs used by docs/tests
```

## Working With Configs

- Validate manually with `pnpm generate` – CLI will surface AJV errors for malformed JSON.
- UI preview shows live JSON + validation messages powered by the shared Zod schema.
- Refer to `docs/configuration.md` for field descriptions.

## Updating Schemas

1. Edit `schemas/config.schema.json`
2. Mirror changes in `app/src/configSchema.ts`
3. Update `docs/configuration.md`
4. Regenerate snapshots/tests: `pnpm test`

## Releasing Artifacts

1. Ensure `pnpm build` and `pnpm build:ui` succeed
2. Tag or publish the dist folder according to release policy (e.g., create a GitHub release)
3. Attach `dist/`, `dist-app/`, and example configs as required

## Troubleshooting

- **Template errors** – confirm JSON schema alignment and that new fields are handled in `templates.ts`
- **UI not reflecting schema changes** – rebuild with `pnpm dev:ui` and check `configSchema.ts`
- **Large bundle warning** – Vite warns if UI bundle exceeds 500 kB; consider code-splitting or adjusting `build.chunkSizeWarningLimit`

## Docker Usage

See `docs/docker.md` for running the generator in containers and publishing images.
