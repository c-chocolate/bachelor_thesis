import json
import re

import pandas as pd
import tldextract as tld


def ngram(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]


def generate_ngram_table(strings):
    table = dict()

    for d in strings:
        for i in range(3,8):
            gram = ngram(d, i)
            for s in gram:
                if s in table:
                    table[s] += 1
                else:
                    table[s] = 1

    return table


def main():
    l = 'list_path_here'
    strings = set()
    with open(l, 'r') as f:
        for domain in f.readlines():
            domain = re.sub(r'^www\d*\.', '', domain)
            ext = tld.extract(domain)
            strings.add(ext.subdomain)
            strings.add(ext.domain)

    table = generate_ngram_table(strings)
    with open('ngram_table2.json', 'w') as f:
        f.write(json.dumps(table))


if __name__ == '__main__':
    main()
