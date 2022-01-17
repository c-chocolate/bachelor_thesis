import re
import sys

import pandas as pd

if len(sys.argv) < 2:
    sys.exit(1)

today = sys.argv[1]

tranco = open(today+'/tranco_100k.txt', 'w')
urlhaus = open(today+'/urlhaus.txt', 'w')
phishtank = open(today+'/phishtank.txt', 'w')

with open(today+'/top-1m.csv', 'r') as f:
    for i in range(100000):
        line = f.readline()
        print(line.rstrip('\n').split(',')[1], file=tranco)
    tranco.close()

df = pd.read_csv(today+'/urlhaus.csv')
#urls = list(df[df['url_status'] == 'online']['url'])
urls = list(df['url'])
for url in urls:
    m = re.match(r'https?://(.*?)/', url+'/')
    if m:
        extract = m.group(1)
        if not extract.replace('.', '').replace(':', '').isdigit():
            print(extract, file=urlhaus)
urlhaus.close()

df = pd.read_csv(today+'/phishtank.csv')
urls = list(df[df['online'] == 'yes']['url'])
for url in urls:
    m = re.match(r'https?://(.*?)/', url+'/')
    if m:
        extract = m.group(1)
        if not extract.replace('.', '').replace(':', '').isdigit():
            print(extract, file=phishtank)
phishtank.close()
