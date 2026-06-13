#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

bash scripts/sync-proposals.sh
python3 scripts/sync-lab-data.py
npm run build

echo "Refreshed proposals, lab snapshot, and site build; build passed."
