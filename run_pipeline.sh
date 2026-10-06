#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data out
NPY="${1:?usage: run_pipeline.sh frame.npy [res_x res_y res_z vel_thresh]}"; shift || true
g++ -std=c++17 -O2 -Wall -Wextra -o obj main.cpp
python3 prepare.py "$NPY" data/points.bin data/index.npy
./obj data/points.bin out/labels.bin "$@"
rm obj
python3 visualize.py data/points.bin out/labels.bin
