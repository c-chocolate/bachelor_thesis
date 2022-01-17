import copy
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd

font_dirs = ['/home/viola/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
# going to be deprecated
font_manager.fontManager.ttflist.extend(font_list)

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'

df = pd.read_csv('imp.csv')

clf = ['lgbm', 'dt']
pattern = ['all', 'top', 'b']

def plot(imp, name, f='plot.png'):
    plt.figure(figsize=(8,8))
    #plt.figure()
    plt.axes([0.3, 0.1, 0.6, 0.85])
    plt.yticks(fontsize=14)
    plt.barh(range(1,26), imp, tick_label=name, log=True)
    plt.xlabel(u'特徴量重要度')
#    plt.ylabel(u'')
    plt.savefig(f, dpi=200)
    plt.close()

name = ['length', 'n_vowel_chars', 'vowel_ratio', 'n_vowels', 'n_constant_chars', 'n_constants', 'vowel_constant_convs', 'n_nums', 'num_ratio', 'alpha_numer_convs', 'n_other_chars', 'max_consecutive_chars', 'entropy', 'n_ip', 'n_mx', 'n_ptr', 'n_ns', 'ns_similarity', 'n_countries', 'mean_TTL', 'stdev_TTL', 'n_labels', 'life_time', 'active_time', 'rv']

translate = {
    'length':'長さ',
    'n_vowel_chars':'母音数',
    'vowel_ratio':'母音割合',
    'n_vowels':'母音種類',
    'n_constant_chars':'子音数',
    'n_constants':'子音種類',
    'vowel_constant_convs':'母音子音の切替',
    'n_nums':'数字数',
    'num_ratio':'数字割合',
    'alpha_numer_convs':'数字アルファベット切替',
    'n_other_chars':'他文字',
    'max_consecutive_chars':'最大繰返し数',
    'entropy':'エントロピー',
    'n_ip':'IP 数',
    'n_mx':'MX 数',
    'n_ptr':'PTR 数',
    'n_ns':'NS 数',
    'ns_similarity':'NS 間の文字列類似度',
    'n_countries':'IP 所属国の数',
    'mean_TTL': 'TTL 平均',
    'stdev_TTL': 'TTL 標準偏差',
    'n_labels': 'HTML タグ数',
    'life_time': 'ライフタイム',
    'active_time': 'アクティブタイム',
    'rv':'評判値'
}


for c in clf:
    for p in pattern:
        for imp in ['imp', 'pimp']:
            data = df[(df['clf'] == c) & (df['pattern'] == p) & (df['imp'] == imp)].iloc[0, 3:].values
            _name = copy.deepcopy(name)
            _data = sorted(zip(data, _name))
            sdata = []
            sname = []
            for d,n in _data:
                sdata.append(d)
                sname.append(translate[n])
            plot(sdata, sname, 'graph_{}_{}_{}.png'.format(c, p, imp))
