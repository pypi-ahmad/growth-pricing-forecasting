#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw/mercari
kaggle competitions download -c mercari-price-suggestion-challenge -p data/raw/mercari
unzip -o data/raw/mercari/mercari-price-suggestion-challenge.zip -d data/raw/mercari
