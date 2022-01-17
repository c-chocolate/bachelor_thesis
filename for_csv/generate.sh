#!/usr/bin/env bash
base_dir="./"
python="./venv/bin/python"
today=$(date +%m%d)

cd "$base_dir"
python　"$base_dir"/generate.py "$today" 2>&1 > "$today".log
