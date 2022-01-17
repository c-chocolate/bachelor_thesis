import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

font = {'family': 'Noto Sans CJK JP'}
matplotlib.rc('font', **font)

df = pd.read_csv('result.csv')

clf = ['lgbm', 'dt']
pattern = ['all', 'top', 'b']
metrics = ['accuracy', 'precision', 'recall', 'f1']

def plot(r, f='plot.png'):
    l = {'accuracy': '精度',
         'precision': '適合率',
         'recall': '再現率',
         'f1': 'F値'}
    plt.figure()
    for m in metrics:
        plt.plot(range(1, 26), r[m], label=l[m])
    plt.xlabel(u'特徴量の数')
    plt.ylabel(u'検知性能')
    plt.legend()
    plt.savefig(f, dpi=200)
    plt.close()

for c in clf:
    for p in pattern:
        for imp in ['imp', 'pimp']:
            r = {}
            for m in metrics:
                r[m] = df[(df['clf'] == c) & (df['pattern'] == p) & (df['imp_type'] == imp) & (df['metric'] == m)].iloc[0, 4:].values
            plot(r, '{}_{}_{}.png'.format(c, p, imp))
