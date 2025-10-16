# syntax=docker/dockerfile:1

FROM node:18-slim AS deps
WORKDIR /workspace
COPY package.json package-lock.json ./
RUN npm ci

FROM node:18-slim AS builder
WORKDIR /workspace
COPY --from=deps /workspace/node_modules ./node_modules
COPY . .
RUN npm run build && npm run build:ui

FROM node:18-slim AS runtime
WORKDIR /workspace
ENV NODE_ENV=production
COPY package.json package-lock.json ./
COPY --from=deps /workspace/node_modules ./node_modules
COPY --from=builder /workspace/dist ./dist
COPY --from=builder /workspace/dist-app ./dist-app
COPY --from=builder /workspace/schemas ./schemas
COPY --from=builder /workspace/docs ./docs
COPY --from=builder /workspace/examples ./examples
COPY --from=builder /workspace/vite.config.ts ./vite.config.ts
COPY --from=builder /workspace/tailwind.config.ts ./tailwind.config.ts
COPY --from=builder /workspace/postcss.config.cjs ./postcss.config.cjs
COPY --from=builder /workspace/tsconfig.json ./tsconfig.json
COPY --from=builder /workspace/README.md ./README.md
COPY docker/entrypoint.sh /usr/local/bin/codex
RUN chmod +x /usr/local/bin/codex

EXPOSE 4173
ENTRYPOINT ["codex"]
CMD ["help"]
