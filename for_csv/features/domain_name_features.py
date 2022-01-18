import math
import re

import tldextract as tld


def entropy(s):
    probas = {i: s.count(i)/len(s) for i in set(s)}
    return -sum((p * math.log2(p)) for p in probas.values())


def domain_name_features(url):
    domain = re.sub(r'^www\d*', '', url.lower()).lstrip('.')
    ext = tld.extract(domain)
    domain = '.'.join(filter(None, [ext.subdomain, ext.domain]))
    # domain = ext.domain

    # feature 1
    length = len(domain)

    if length == 0:
        return 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0

    # feature 2_1
    n_vowel_chars = 0
    # feature 3_1
    n_constant_chars = 0
    # feature 5
    n_nums = 0

    n_other_chars = 0

    vowel_set = set()
    constant_set = set()

    # feature 4
    vowel_constant_convs = 0
    # feature 7
    alpha_numer_convs = 0

    prev_is_vowel = False
    prev_is_constant = False
    prev_is_num = False

    match = re.findall(r'(([A-Za-z\-])\2*)', domain)
    if match:
        max_consecutive_chars = max([len(group[0]) for group in match])
    else:
        max_consecutive_chars = 0

    dot_count = 0

    for c in domain:
        if c in 'aeiou':
            if prev_is_constant:
                vowel_constant_convs += 1
            elif prev_is_num:
                alpha_numer_convs += 1

            n_vowel_chars += 1
            vowel_set.add(c)
            prev_is_constant = False
            prev_is_num = False
            prev_is_vowel = True
        elif c in 'bcdfghjklmnpqrstvwxyz':
            if prev_is_vowel:
                vowel_constant_convs += 1
            elif prev_is_num:
                alpha_numer_convs += 1

            n_constant_chars += 1
            constant_set.add(c)
            prev_is_vowel = False
            prev_is_num = False
            prev_is_constant = True
        elif c in '0123456789':
            if prev_is_vowel or prev_is_constant:
                alpha_numer_convs += 1
            n_nums += 1
            prev_is_vowel = False
            prev_is_constant = False
            prev_is_num = True
        else:
            if c == '.':
                dot_count += 1
            else:
                n_other_chars += 1

    # feature 2_2
    vowel_ratio = n_vowel_chars / length
    # feature 6
    num_ratio = n_nums / length
    # feature 2_3
    n_vowels = len(vowel_set)
    # feature 3_2
    n_constants = len(constant_set)

    ent = entropy(domain)

    try:
        return (length,
            n_vowel_chars, vowel_ratio, n_vowels,
            n_constant_chars, n_constants,
            vowel_constant_convs / (length - dot_count - 1),
            n_nums,
            num_ratio,
            alpha_numer_convs / (length - dot_count - 1),
            n_other_chars,
            max_consecutive_chars,
            ent)
    except ZeroDivisionError:
        return (length,
            n_vowel_chars, vowel_ratio, n_vowels,
            n_constant_chars, n_constants,
            0.0,
            n_nums,
            num_ratio,
            0.0,
            n_other_chars,
            max_consecutive_chars,
            ent)
        


if __name__ == '__main__':
    domain = 'zmfcgxwchmkfvqrwnnmgbvrsqjtcfwxr.soho.limo'
    #print(domain_name_features(domain))
    print(entropy('cg79wo20kl92doowfn01oqpo9mdieowv5tyj'))
    print(domain_name_features('www-infosec.ist.osaka-u.ac.jp'))
