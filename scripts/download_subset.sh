#!/usr/bin/env bash
# Download a ds004148 subset for EEG Mood-Sync.
# Usage: bash scripts/download_subset.sh [target_dir]

set -euo pipefail

TARGET="${1:-data/ds004148}"
mkdir -p "${TARGET}"

if ! command -v openneuro-py >/dev/null 2>&1; then
  echo "openneuro-py not found. Install: pip install openneuro-py"
  exit 1
fi

# Download full subject folders (3 sessions × 5 tasks each).
# Avoid wildcards — openneuro-py may reject them.
for sub in 01 02 03 04 05; do
  echo "=== Downloading sub-${sub} (all sessions) ==="
  openneuro-py download \
    --dataset ds004148 \
    --target_dir "${TARGET}" \
    --include "sub-${sub}"
done

echo "Done. Example check:"
echo "ls ${TARGET}/sub-01/ses-session1/eeg/"
