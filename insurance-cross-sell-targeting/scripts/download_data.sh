#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/raw/insurance_cross_sell
kaggle competitions download -c playground-series-s4e7 -p data/raw/insurance_cross_sell
unzip -o data/raw/insurance_cross_sell/playground-series-s4e7.zip -d data/raw/insurance_cross_sell
