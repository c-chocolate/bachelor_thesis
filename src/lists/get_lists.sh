#!/usr/bin/env bash

base_dir="./"
today=$(date +%m%d)
tranco="https://tranco-list.eu/top-1m.csv.zip"
urlhaus="https://urlhaus.abuse.ch/downloads/csv_recent/"
phishtank="http://data.phishtank.com/data/api-key-here/online-valid.csv"
cybercrime="https://cybercrime-tracker.net/all.php"

mkdir -p $base_dir

if [[ -d "$base_dir/$today" ]]; then
	exit 0
fi

mkdir -p "$base_dir/$today"
cd "$base_dir/$today"

curl -L "$tranco" -o top-1m.csv.zip
unzip top-1m.csv.zip
rm -f top-1m.csv.zip

curl -L "$urlhaus" -o urlhaus.tmp
cat urlhaus.tmp | sed 's/^#\ \(id.*\)/\1/' | sed '/^#.*$/d' > urlhaus.csv
rm -f urlhaus.tmp

curl -L "$phishtank" -o phishtank.csv

curl -L "$cybercrime" -o cybercrime.tmp
cat cybercrime.tmp | sed 's/\/.*//' | sed '/^[0-9]\{1,3\}\.[0-9]\{1,3\}.[0-9]\{1,3\}.[0-9]\{1,3\}/d' > cybercrime.txt
rm -f cybercrime.tmp

cd ..
python3 process.py "$today"

cd "$today"
cat phishtank.txt urlhaus.txt cybercrime.txt | sed '/^[[:space:]]*$/d' | sort | uniq > blacklist.txt
