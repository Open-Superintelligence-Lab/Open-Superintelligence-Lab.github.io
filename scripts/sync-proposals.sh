#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="/Users/vukrosic/my-life/orchestrator/proposals"
TARGET_DIR="${REPO_ROOT}/proposals"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source proposals directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
find "${TARGET_DIR}" -maxdepth 1 -type f -name '*.md' -delete

shopt -s nullglob
for file in "${SOURCE_DIR}"/*.md; do
  cp "${file}" "${TARGET_DIR}/"
done

echo "Synced proposals from ${SOURCE_DIR} to ${TARGET_DIR}"
