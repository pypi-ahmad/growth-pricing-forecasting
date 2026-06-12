#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw/rossmann
kaggle competitions download -c rossmann-store-sales -p data/raw/rossmann
unzip -o data/raw/rossmann/rossmann-store-sales.zip -d data/raw/rossmann
