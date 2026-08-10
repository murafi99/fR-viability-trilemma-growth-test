#!/usr/bin/env bash
# Regenerates every table and figure in the paper from source.
set -e
cd "$(dirname "$0")"
mkdir -p figures results
python -m src.fit_data
python -m src.make_figures
python -m src.background
python -m src.screening
echo "Done. See results/ and figures/."
