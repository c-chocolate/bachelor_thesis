#!/usr/bin/env bash

#features/doc2vec.modelが必要
#features/en_to_ja.txtが必要
#blacklist.txtが必要 あらかじめ数を変更
#tranco_100k.txtが必要
base_dir="./"
python="./venv/bin/python"
today=$(date +%m%d)

#cd "$base_dir"
python　"$base_dir"/generate.py "$today" 2>&1 > "$today".log
