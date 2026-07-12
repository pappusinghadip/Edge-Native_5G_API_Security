#!/bin/sh
set -eu

python -m src.fl.client --config configs/fl.yaml --paths configs/paths.yaml
