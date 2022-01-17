import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier as lgbmc
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import normalize


seed = 5600

clean = pd.read_csv('0_1125.csv').fillna(0)
malware = pd.read_csv('1_1125.csv') 

data = pd.concat([clean, malware], ignore_index=True)
X = data.iloc[:, 1:]
y = data.iloc[:, :1].values.flatten()

top = pd.concat([clean[:len(malware)], malware], ignore_index=True)
X_top = top.iloc[:, 1:]
y_top = top.iloc[:, :1].values.flatten()

balanced = pd.concat([clean.sample(n=len(malware), random_state=seed), malware], ignore_index=True)
X_b = balanced.iloc[:, 1:]
y_b = balanced.iloc[:, :1].values.flatten()

del data, top, balanced

dt = DT(max_depth=10, random_state=seed)
lgbm = lgbmc(objective='binary', importance_type='gain', random_state=seed, n_estimators=400, num_leaves=127)


patterns = {'all': [X, y], 'top': [X_top, y_top], 'b': [X_b, y_b]}
clfs = {'lgbm': lgbm, 'dt': dt}
metrics = ['accuracy', 'precision', 'recall', 'f1']


fp = open('result.csv', 'w')

for name, clf in clfs.items():
    for pname, p in patterns.items():
        print('running {} {}'.format(name, pname))
        clf.fit(p[0], p[1])
        imp = normalize([clf.feature_importances_])[0]
        print('imp ok')
        pimp = normalize([permutation_importance(clf, p[0], p[1],
                                                 n_repeats=5, random_state=seed).importances_mean])[0]
        print('pimp ok')
# save imp and pimp here to generate imp.csv

        print('generating imp')
        results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for i in range(25):
            print(i)
            ft_imp = [x for _, x in sorted(zip(imp, X.columns), reverse=True)[:i+1]]
            for m in metrics:
                results[m].append(np.mean(cross_val_score(clf, p[0].loc[:, ft_imp], p[1],
                                                          n_jobs=-1, scoring=m, cv=StratifiedKFold(10))))
        for k, v in results:
            print('{},{},{},{},{}'.format(name, pname, i, k, ','.join(v)),
                file=fp)

        print('generating pimp')
        results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for i in range(25):
            print(i)
            ft_pimp = [x for _, x in sorted(zip(pimp, X.columns), reverse=True)[:i+1]]
            for m in metrics:
                results[m].append(np.mean(cross_val_score(clf, p[0].loc[:, ft_pimp], p[1],
                                                          n_jobs=-1, scoring=m, cv=StratifiedKFold(10))))
        for k, v in results:
            print('{},{},{},{},{}'.format(name, pname, i, k, ','.join(v)),
                file=fp)

close(fp)
