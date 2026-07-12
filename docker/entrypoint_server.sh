#!/bin/sh
set -eu

python -m src.fl.server --config configs/fl.yaml --paths configs/paths.yaml
