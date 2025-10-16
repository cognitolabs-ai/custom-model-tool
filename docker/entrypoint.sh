#!/usr/bin/env bash
set -euo pipefail

command="${1:-help}"
shift || true

case "$command" in
  help|-h|--help)
    cat <<'EOF'
Codex Notebook Generator container

Usage:
  codex cli [args...]    Run the notebook generator CLI (node dist/cli.js)
  codex ui [args...]     Serve the pre-built UI with vite preview (uses $PORT or 4173)
  codex help             Show this message
  codex <cmd>            Execute any other command

Examples:
  codex cli --config examples/configs/full.json --out tmp/run_full
  PORT=8080 codex ui
EOF
    ;;
  cli|generate)
    exec node dist/cli.js "$@"
    ;;
  ui)
    port="${PORT:-4173}"
    exec npx vite preview --config vite.config.ts --host 0.0.0.0 --port "$port" "$@"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac

