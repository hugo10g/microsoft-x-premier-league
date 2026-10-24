#!/usr/bin/env bash
# Build the dashboard image without Docker and push it to Azure Container Registry.
#
#   scripts/push_image.sh <registry-name> <tag>      e.g. scripts/push_image.sh offballvalue v2
#
# Azure for Students blocks ACR Tasks (cloud builds) and this machine has no Docker, so
# the image is assembled from layers: python:3.13-slim, the Linux wheels of
# requirements.txt, and the app with the league and its precomputed scores
# (run scripts/precompute_scores.py first). It mirrors the Dockerfile, which GitHub
# Actions builds. Needs: az signed in (or $AZ), crane (github.com/google/go-containerregistry)
# on PATH or in $CRANE, and the project's .venv. $SUBSCRIPTION defaults to "Azure for Students".
set -euo pipefail

registry=${1:?registry name}; tag=${2:?tag}
crane=${CRANE:-crane}; az=${AZ:-az}; subscription=${SUBSCRIPTION:-Azure for Students}
server="$registry.azurecr.io"
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

[[ -f "$root/data/synthetic/scored.pkl" ]] || { echo "Run scripts/precompute_scores.py first." >&2; exit 1; }

echo "Linux wheels..."
grep -vE '^(pytest|esprima)' "$root/requirements.txt" > "$work/requirements.txt"
"$root/.venv/bin/pip" install -q --disable-pip-version-check --only-binary=:all: \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --implementation cp --python-version 3.13 --abi cp313 \
    --target "$work/deps/usr/local/lib/python3.13/site-packages" -r "$work/requirements.txt"

echo "Layers..."
pack() { COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --no-mac-metadata --no-xattrs -cf "$@"; }
pack "$work/deps.tar" -C "$work/deps" usr
(cd "$root" && pack "$work/app.tar" -s ',^\.,app,' --exclude ./.env --exclude ./.git --exclude ./.venv \
    --exclude ./outputs --exclude __pycache__ --exclude .pytest_cache --exclude .DS_Store .)
tar -tf "$work/app.tar" | grep -qx 'app/.env' && { echo "Refusing: .env would be in the image." >&2; exit 1; }

echo "Push to $server..."
"$az" acr login --subscription "$subscription" -n "$registry" --expose-token --query accessToken -o tsv --only-show-errors \
    | "$crane" auth login "$server" -u 00000000-0000-0000-0000-000000000000 --password-stdin >/dev/null
"$crane" append --platform linux/amd64 -b python:3.13-slim -f "$work/deps.tar" -f "$work/app.tar" \
    -t "$server/off-ball-value:$tag-base" >/dev/null
"$crane" mutate "$server/off-ball-value:$tag-base" --workdir /app --exposed-ports 8501 \
    --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
    --cmd "python,-m,streamlit,run,app.py,--server.address=0.0.0.0,--server.port=8501,--server.headless=true,--browser.gatherUsageStats=false" \
    -t "$server/off-ball-value:$tag"
"$crane" auth logout "$server" >/dev/null
echo "Pushed $server/off-ball-value:$tag"
