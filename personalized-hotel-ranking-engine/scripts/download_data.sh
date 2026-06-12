#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw/expedia
kaggle competitions download -c expedia-hotel-recommendations -p data/raw/expedia
unzip -o data/raw/expedia/expedia-hotel-recommendations.zip -d data/raw/expedia
