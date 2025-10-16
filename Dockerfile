# syntax=docker/dockerfile:1

FROM node:18-slim AS deps
WORKDIR /workspace
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

FROM deps AS builder
COPY . .
RUN npm run build && npm run build:ui

FROM node:18-slim AS runtime
WORKDIR /workspace
ENV NODE_ENV=production
ENV PORT=4173
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
COPY --from=builder /workspace/dist ./dist
COPY --from=builder /workspace/dist-app ./dist-app
COPY --from=builder /workspace/schemas ./schemas
COPY --from=builder /workspace/docs ./docs
COPY --from=builder /workspace/examples ./examples
COPY docker/entrypoint.sh /usr/local/bin/codex
RUN chmod +x /usr/local/bin/codex

EXPOSE 4173
ENTRYPOINT ["codex"]
CMD ["help"]
