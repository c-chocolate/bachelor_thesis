import json
import math
import os.path

import tldextract as tld


packagedir = os.path.dirname(__file__)
ngram_table = None
with open(os.path.join(packagedir, 'ngram_table.json')) as f:
    ngram_table = json.loads(f.readline())


def ngram(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]


def extra_features(domain):
    ext = tld.extract(domain)    

    rep = 0.0
    for s in [ext.subdomain, ext.domain]:
        for i in range(3,8):
            grams = ngram(s, i)
            for gram in grams:
                if gram in ngram_table:
                    rep += math.log2(ngram_table[gram] / i)

    return [rep]

if __name__ == '__main__':
    print(extra_features('www-infosec.ist.osaka-u.ac.jp'))
