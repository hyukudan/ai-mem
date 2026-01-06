#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/install-hooks.sh [--dest PATH] [--bin PATH] [--project PATH] [--write-env]

Installs model-agnostic hook scripts to a shared folder for reuse.

Options:
  --dest PATH     Destination directory (default: $AI_MEM_HOOKS_DIR or ~/.config/ai-mem/hooks)
  --bin PATH      ai-mem binary path to store in hooks.env (optional)
  --project PATH  Default project path to store in hooks.env (optional)
  --write-env     Write hooks.env (overwrites if exists)
EOF
}

dest_override=""
bin_path=""
project_path=""
write_env="false"

while [ $# -gt 0 ]; do
  case "$1" in
    --dest)
      dest_override="${2:-}"
      shift 2
      ;;
    --bin)
      bin_path="${2:-}"
      shift 2
      ;;
    --project)
      project_path="${2:-}"
      shift 2
      ;;
    --write-env)
      write_env="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

dest="${dest_override:-${AI_MEM_HOOKS_DIR:-$HOME/.config/ai-mem/hooks}}"
mkdir -p "$dest"

cp -R "scripts/hooks/." "$dest/"
chmod +x "$dest"/*.sh 2>/dev/null || true

echo "Installed hooks to $dest"

example_env="$dest/hooks.env.example"
cat <<EOF > "$example_env"
# Example env file for ai-mem hooks.
# Source this file in your shell, or set the variables in your hook runner.

# AI_MEM_BIN="$bin_path"
# AI_MEM_PROJECT="$project_path"
# AI_MEM_SESSION_TRACKING=1
# AI_MEM_SUMMARY_ON_END=1
EOF
echo "Wrote $example_env"

if [ "$write_env" = "true" ]; then
  env_path="$dest/hooks.env"
  cat <<EOF > "$env_path"
AI_MEM_BIN="${bin_path}"
AI_MEM_PROJECT="${project_path}"
EOF
  echo "Wrote $env_path"
fi
