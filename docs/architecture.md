# Architectural Overview

## System Context

Codex Notebook Generator produces fine-tuning bundles (notebook, config, README, ZIP) for large language model workflows. The codebase consists of three cooperating packages:

- **Core generator (`src/`)** – TypeScript modules that validate input configs, render notebooks/sidecars, and package artifacts.
- **Command Line Interface (`src/cli.ts`)** – Thin wrapper around the core generator that reads a config file and writes outputs to disk.
- **Browser UI (`app/`)** – React single-page app that lets practitioners compose configurations visually and download artifacts without leaving the browser.

All packages are built with Node.js 18+, TypeScript, and pnpm. The generator is publishable as a library, while the UI may be hosted as static assets or run through `vite preview`.

## Component Diagram

```
┌───────────────┐          ┌────────────────────────┐
│   Config JSON │──validate──▶ renderBundle.ts (AJV, │
└───────────────┘          │ Nunjucks, JSZip)      │
                            └──────┬───────────────┘
                                   │ embeds config, builds notebook
                                   ▼
                            ┌───────────────┐
                            │ templates.ts  │  (nbformat template + markdown)
                            └──────┬────────┘
                                   │ JSON notebook, YAML config, README
                                   ▼
                            ┌───────────────┐
                            │  generate     │
                            │  ZIP + files  │
                            └───────────────┘
```

The UI imports the same `renderBundle` function (via Vite aliases) so browser and CLI outputs stay consistent.

## Core Flow

1. **Validation** – `renderBundle.ts` uses AJV and `schemas/config.schema.json` to ensure the JSON payload matches supported parameters (including accelerator, training mode, RL sections, compile/profiler blocks).
2. **Templating** – Nunjucks renders Notebook/README/Config templates (TypeScript typed). The notebook JSON is parsed and the config cell is injected with timestamped payload.
3. **Packaging** – JSZip optionally creates a bundle, while individual files are returned to callers. CLI writes to disk; UI streams downloads through `Blob` URLs.
4. **Extensions** – Template logic conditionally enables Unsloth acceleration, TRL trainers, `torch.compile`, and `torch.profiler` instrumentation based on config flags.

## UI Architecture

- **State** – A single `UiConfig` state tree mirrors the JSON schema. Zod validation (`app/src/configSchema.ts`) powers inline validation messages and drives the JSON preview.
- **Sections** – UI is grouped into Model & Strategy, Dataset, Hyperparameters, RL settings, HRM overlay, Logging/Outputs, compile/profiler toggles, and metrics.
- **Downloads** – After calling `renderBundle`, the UI creates `Blob` URLs for each artifact and offers per-file download buttons plus a ZIP bundle when available.
- **Styling** – Tailwind CSS (configured via `tailwind.config.ts`) and the Inter font supply the design system.

## Build Outputs

| Target        | Tool     | Output directory      | Purpose                         |
|---------------|----------|-----------------------|---------------------------------|
| Core library  | tsup     | `dist/`                | Node-friendly JS for CLI / libs |
| CLI           | tsup     | `dist/cli` (via tsup)  | Executable entry point          |
| Browser UI    | Vite     | `dist-app/`            | Production-ready static assets  |

During CI and Docker builds we run `pnpm install`, `pnpm build`, and `pnpm build:ui` to supply both Node and browser artifacts.

## Testing Strategy

- **Unit & snapshot tests** – `pnpm test` executes Vitest suites covering schema validation, template rendering, and notebook snapshots.
- **UI smoke** – Vite’s type-checking and build steps catch JSX/TSX issues; integration tests can be added with Playwright.
- **Schema discipline** – Changes to `config.schema.json` must be duplicated in `app/src/configSchema.ts` to keep UI validation in sync; see `docs/operations.md`.

## Key Modules

| File | Responsibility |
|------|----------------|
| `src/core/renderBundle.ts` | Validation + template rendering + zip packaging |
| `src/core/templates.ts` | nbformat template, README template, YAML config template |
| `src/generateNotebook.ts` | Writes rendered artifacts to disk and returns paths |
| `src/cli.ts` | Parses CLI flags and invokes generator |
| `app/src/App.tsx` | Main React app with Tailwind UI |
| `app/src/configSchema.ts` | Zod schema mirroring JSON schema |

## Extensibility Notes

- **Templates** – To add new notebook sections, update `templates.ts`, ensure they handle conditional config values, and refresh snapshot tests.
- **Schema** – Add new fields first to `schemas/config.schema.json`, mirror in `configSchema.ts`, and bump docs (see `docs/configuration.md`).
- **Integrations** – Hooks for additional accelerators or RL trainers should piggyback on existing conditional branches in `templates.ts` and UI toggles.

## Future Enhancements

- Automated notebook execution tests (e.g., Papermill) in CI.
- Breaking out shared config typings into a dedicated package for downstream usage.
- Live schema documentation generation to avoid manual duplication.
