import os.path
import sys

from features import domain_name_features, dns_features, contextual_features
from features import extra_features
from joblib import Parallel, delayed

from features import add_web_features   #追加

#output_num_print=0

if len(sys.argv) < 2:
    print('Usage: {} DATE'.format(sys.argv[0]))
    sys.exit(1)

date = sys.argv[1]
for i in range(2):
    os.makedirs('{}_{}'.format(i, date), exist_ok=True)

def generate(label, index, url):
    filename = './{}_{}/{}.csv'.format(label, date, index+1)
    if os.path.isfile(filename) and os.path.getsize(filename) > 40:
        return

    features = [label]
    features.extend(domain_name_features(url))
    features.extend(dns_features(url))
    features.extend(contextual_features(url))
    features.extend(extra_features(url))
    features.extend(add_web_features(url))      #追加

    print(','.join(format(i, 'g') for i in features), file=open(filename, 'w'))
    #output_num_print+=1     #追加
    #print("NOW",output_num_print)   #追加
    

arg_list = []
count = []
#with open('lists/{}/tranco_100k.txt'.format(date), 'r') as f:
with open('test.txt'.format(date), 'r') as f:   #追加
    lines = f.readlines()
    urls = list(map(lambda it: it.rstrip('\n'), lines))
    count.append(len(urls))
    arg_list.extend([(0, i, urls[i]) for i in range(len(urls))])
#with open('lists/{}/blacklist.txt'.format(date), 'r') as f:
with open('test2.txt'.format(date), 'r') as f:      #追加
    lines = f.readlines()
    urls = list(map(lambda it: it.rstrip('\n'), lines))
    count.append(len(urls))
    arg_list.extend([(1, i, urls[i]) for i in range(len(urls))])

del lines
del urls

#Parallel(n_jobs=96, verbose=3)([delayed(generate)(*arg) for arg in arg_list])
Parallel(n_jobs=50, verbose=3)([delayed(generate)(*arg) for arg in arg_list])

columns = [
    'label', 'length',
    'n_vowel_chars', 'vowel_ratio', 'n_vowels',
    'n_constant_chars', 'n_constants', 'vowel_constant_convs',
    'n_nums', 'num_ratio', 'alpha_numer_convs',
    'n_other_chars', 'max_consecutive_chars', 'entropy',
    'n_ip', 'n_mx', 'n_ptr', 'n_ns', 'ns_similarity',
    'n_countries', 'mean_TTL', 'stdev_TTL',
    'n_labels', 'life_time', 'active_time',
    'rv',
    'n_css_selectors', 'html_id_class_num', 'get_html_id_class_rate',       #追加
    'n_js_function', 'js_comparison_average', 'js_comparison_max', 'js_comparison_min']     #追加

for label in range(2):
    with open('{}_{}.csv'.format(label, date), 'w') as f:
        print(','.join(columns), file=f)
        for i in range(count[label]):
            filename = './{}_{}/{}.csv'.format(label, date, i+1)
            if os.path.isfile(filename) and os.path.getsize(filename) > 40:
                with open(filename, 'r') as g:
                    line = g.readline().rstrip('\n')
            print(line, file=f)
