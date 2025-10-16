# Docker Guide

This project ships with a multi-stage Dockerfile that produces an image capable of running the CLI and serving the React UI.

## Building the Image

```bash
docker build -t codex-notebook-generator .
```

The build pipeline performs the following:

1. Restores npm dependencies (`npm ci`).
2. Compiles the TypeScript core (`npm run build`).
3. Builds the production UI bundle (`npm run build:ui`).
4. Copies compiled artifacts into a slim runtime image alongside the shared schema, docs, and examples.

## Entrypoint Commands

The container entrypoint script (`codex`) accepts multiple subcommands:

| Command | Description |
|---------|-------------|
| `cli [args]` | Executes the compiled CLI (`node dist/cli.js`). Pass standard flags such as `--config` and `--out`. |
| `ui [args]` | Serves the pre-built Vite bundle using `vite preview`. Listens on `$PORT` (defaults to `4173`). |
| anything else | Executed verbatim (e.g., `bash`, `node`). |

## CLI Example

```bash
docker run --rm \
  -v "$PWD/examples:/workspace/examples" \
  -v "$PWD/tmp:/workspace/tmp" \
  codex-notebook-generator \
  cli --config examples/configs/full.json --out tmp/run_full
```

## UI Preview Example

```bash
docker run --rm -p 4173:4173 codex-notebook-generator ui
```

You can override the port with `-e PORT=8080`.

## Notes

- The image retains `node_modules` (including dev dependencies) to support Vite preview and additional tooling.
- `schemas/` is bundled so runtime validation works without accessing the source tree.
- Artifacts are generated under `/workspace` inside the container. Mount a volume to retrieve outputs.
- For production hosting of the UI, consider copying `dist-app/` into a dedicated static web server image (e.g., Nginx) rather than using `vite preview`.

## Extending the Image

- Add environment variables or secrets using standard Docker run flags (`-e` / `--env-file`).
- To install extra Python tooling for notebook post-processing, extend the final stage with additional package installs.
- CI builds (see `.github/workflows/ci.yml`) can push the image to your container registry by enabling the `DOCKERHUB_*` or `GHCR_*` secrets.

## Publishing to GitHub Container Registry (GHCR)

This repository includes a GitHub Actions workflow that builds and pushes the Docker image to GHCR on tag/release events.

- Image name: `ghcr.io/<owner>/<repo>:<tag>` (for this repo, `ghcr.io/${OWNER}/${REPO}`)
- Tags: semver (`vX.Y.Z`, `X.Y`, `X`), `sha` and the release tag itself

### Releasing

1. Create a tag: `git tag v0.2.0 && git push origin v0.2.0`
2. Or publish a GitHub Release from the UI
3. GitHub Actions builds and pushes to GHCR automatically

### Pulling

```bash
# authenticate (optional for public repos)
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-username> --password-stdin

# pull the image
docker pull ghcr.io/<owner>/<repo>:v0.2.0

# run CLI in the container
docker run --rm -v "$PWD/examples:/workspace/examples" -v "$PWD/tmp:/workspace/tmp" ghcr.io/<owner>/<repo>:v0.2.0 \
  cli --config examples/configs/full.json --out tmp/run_full

# run UI preview
docker run --rm -p 4173:4173 ghcr.io/<owner>/<repo>:v0.2.0 ui
```
## Docker Compose (Development Helpers)

A `docker-compose.yml` file is provided for convenience:

- `cli` service runs the generator once against the sample config (writes outputs to `./tmp`).
- `ui` service serves the bundled React UI on port 4173.

```bash
# Start the UI preview (and build the image if needed)
docker compose up ui

# Run the CLI job (one-shot)
docker compose run --rm cli
```

Both services reuse the same Dockerfile; feel free to extend the compose file with your own volumes or environment variables.

