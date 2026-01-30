#!/usr/bin/env bash
set -euo pipefail

# Removes local artifacts that should not be uploaded if you plan to share a zip.
# Safe by default: requires --yes to actually delete.

if [[ "${1:-}" != "--yes" ]]; then
  echo "Dry run. To actually delete, re-run:"
  echo "  scripts/clean_for_github.sh --yes"
  echo
  echo "Would remove:"
  echo "  .venv/"
  echo "  data/raw/  data/interim/  data/processed/  data/cache/"
  echo "  reports/"
  echo "  model-studio/node_modules/"
  echo "  model-studio/public/model_data.json"
  exit 0
fi

rm -rf .venv
rm -rf data/raw data/interim data/processed data/cache
rm -rf reports
rm -rf model-studio/node_modules
rm -f model-studio/public/model_data.json

echo "Done."

