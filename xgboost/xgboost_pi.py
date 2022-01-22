import xgboost as xgb
import pandas as pd
import numpy as np
#from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold       # データをトレーニングセットとテストセットに分割するためのトレーニング/テストインデックスを提供
from art.utils import to_categorical        # ラベルの配列をバイナリクラスマトリックスに変換
from sklearn.preprocessing import StandardScaler        # 平均値を取り除き、単位分散に合わせてスケーリングすることで、特徴を標準化
import math     # 数学関数
import sys      # 変数　関数
import argparse     # コマンドライン引数
import os       # os依存のモジュール
#from elm_model import ExtremeLearningMachine        # ELM
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score     # 特定の目的のために予測誤差を評価する関数を実装
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

# プログラムの引数の対応
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", default=10, type=int)      # -t 特徴量の数
#parser.add_argument("-n", "--n_unit", default=600, type=int)        # -n 中間ノード数
#parser.add_argument("-f", "--file_name_path", default="file_name_path", type=str)       # -f ?? 出力ファイル名
args = parser.parse_args()

threshold = args.threshold          # -t
#n_unit = args.n_unit                # -n
#file_name = args.file_name_path     # -f

def extract_data_balanced(df):      #l190
    benign = df["label"] == 0       # benign:良性  benign:boolean型  csvのlabel列が0だったらTrue
    benign[4001:] = False      # 24127から最後まで　24127:収集したドメインの数     #変更
    df = pd.concat([df.loc[benign, :], df.loc[df["label"] == 1, :]], axis=0)        # concat:連結　axis=0:列に沿った処理　df.loc:配列から値取ってくるみたいな処理
    return df    #24127ずつ良性と悪性を取ってくる

def get_average_result(perm_list):      #l130
    result = pd.DataFrame(      #二次元の表形式データ
        data=0,     #DataFrameに格納するデータを配列などで指定
        columns=["feature_importance"],        #列名
        index=list(columns_dic.keys()),         #行名
    )
    for pi in perm_list:        #result表形式データにperm_list(引数)を入れていく
        #print(pi)
        result += pi
    result = result / len(perm_list)        #??len
    print(result)
    return result


def output_perm_imp(perm_list, columns_dic):        #l294
    result = get_average_result(perm_list)      #l82の関数
    perm_imp_df = pd.DataFrame(
        {
            "feature_importance": result["feature_importance"],     #??列名をどうするのか　l85のところ l109と同じ
        },
        index=list(columns_dic.keys()),
    )
    print(perm_imp_df)
    perm_imp_df = perm_imp_df.sort_values("feature_importance", ascending=False)      #accendingFalse:降順
    perm_imp_df.to_csv("./data/perm_imp.csv")        #csv出力ファイル名、場所指定

#############
testmode=0
mode=2      #1:ranking 2:threshold
#############

file_name="dataset_0118.csv"

# load data
df = pd.read_csv(file_name)     #csvファイルを読み込んでデータフレーム化
# Replace nan
df = df.replace(np.nan, 0)      #欠損値(nan)を0に置き換える

df = extract_data_balanced(df)

columns_dic = {
    column: index for index, column in enumerate(df.drop("label", axis=1).columns)      #df.drop:行を削除   axis1:行　.colums列を取ってくる　   特徴量の名前とインデックスのdict
}

if mode==1:
    features =list(columns_dic.keys())  #.keys:dictの値を取得   colums_dicのcolumnを取ってくる
elif mode==2:
    perm_imp_df = pd.read_csv("./data/perm_imp.csv", index_col=0)
    features = list(perm_imp_df.index)[:threshold]



df = df.loc[:, features + ["label"]]        #df.loc:pandasの要素抽出(下限から)      featuresで選んだ特徴量とlabelからなるdfに上書き
x_data = df.drop("label", axis=1).values        #valuesでdfをndarray化　label以外
y_data = df["label"].values #labelだけ

FOLD_NUM = 5        #KFoldの分割数
fold_seed = 227      #乱数のシード指定
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)      #Kfold:K-分割交差検証(データをn_split個に分け、n個を訓練用、k-n個をテスト用として使う)
fold_iter = folds.split(x_data)     #x_dataを区切り文字で分割   5個に分割




if mode==1:
    perm_list = []

    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
        print(f"Fold times:{n_fold}")
        x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
        y_train, y_test = y_data[trn_idx], y_data[val_idx]

        # xgboostモデルの作成
        model = xgb.XGBClassifier(verbosity= 0)       #verbosity??

        # ハイパーパラメータ探索
        model_cv = GridSearchCV(model, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
        model_cv.fit(x_train, y_train)
        print(model_cv.best_params_, model_cv.best_score_)

        # 改めて最適パラメータで学習
        model = xgb.XGBClassifier(**model_cv.best_params_)
        model.fit(x_train, y_train)

        fti = model.feature_importances_  

        #print('Feature Importances:')
        #for i, feat in enumerate(features):
        #    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
        perm_imp_df = pd.DataFrame(     #二次元の表形式データ
            {
            "feature_importance": fti,     #平均
            },
            index=list(columns_dic.keys()),     #特徴量が行
        )
        perm_list.append(perm_imp_df)       #リストに要素を追加
        print(perm_imp_df)


    output_perm_imp(perm_list, columns_dic)


if mode==2:
    # acc,precision,recall,f1 train
    acc_train_total = []        #訓練用
    precision_train_total = []
    recall_train_total = []
    f1_train_total = []
    # acc,precision,recall,f1 test
    acc_test_total = []     #実行用
    precision_test_total = []
    recall_test_total = []
    f1_test_total = []
    eval_result = {}        #実験結果用

    print(x_data[0])
    print(y_data[0])
    print(features)
    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
        print(f"Fold times:{n_fold}")
        x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
        y_train, y_test = y_data[trn_idx], y_data[val_idx]

        # xgboostモデルの作成
        clf = xgb.XGBClassifier(verbosity= 0)       #verbosity??

        # ハイパーパラメータ探索
        clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
        clf_cv.fit(x_train, y_train)
        #print("1")
        #print(clf_cv.best_params_, clf_cv.best_score_)

        # 改めて最適パラメータで学習
        clf = xgb.XGBClassifier(**clf_cv.best_params_)
        clf.fit(x_train, y_train)

        y_true_train = y_train
        y_pred_train = clf.predict(x_train)
        y_true_test = y_test
        y_pred_test = clf.predict(x_test)
    
        # evaluate train 
        acc_train_total.append(accuracy_score(y_true_train, y_pred_train))      #結果を格納して追加していってる
        precision_train_total.append(precision_score(y_true_train, y_pred_train))
        recall_train_total.append(recall_score(y_true_train, y_pred_train))
        f1_train_total.append(f1_score(y_true_train, y_pred_train))
        # evaluate test     
        acc_test_total.append(accuracy_score(y_true_test, y_pred_test))
        precision_test_total.append(precision_score(y_true_test, y_pred_test))
        recall_test_total.append(recall_score(y_true_test, y_pred_test))
        f1_test_total.append(f1_score(y_true_test, y_pred_test))
        #print(classification_report(y_test, y_pred_test))
  

    eval_result["train_accuracy"] = np.average(acc_train_total)     #5つの平均計算
    eval_result["train_precision"] = np.average(precision_train_total)
    eval_result["train_recall"] = np.average(recall_train_total)
    eval_result["train_f1"] = np.average(f1_train_total)

    eval_result["test_accuracy"] = np.average(acc_test_total)
    eval_result["test_precision"] = np.average(precision_test_total)
    eval_result["test_recall"] = np.average(recall_test_total)
    eval_result["test_f1"] = np.average(f1_test_total)

    eval_df = pd.DataFrame.from_dict(eval_result, orient="index")       #from_dict:辞書型からDataFrameの生成
    eval_df = eval_df.rename(columns={0: threshold})        #行0をthresholdに変更

    # Output results
    eval_df.to_csv(
        "./output/eval_threshold{}.csv".format(
            threshold
        )
    )




if testmode==1:
    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
        print(f"Fold times:{n_fold}")
        x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
        y_train, y_test = y_data[trn_idx], y_data[val_idx]
        #x_train_std, x_test_std, _ = standard_trans(x_train, x_test)    #l26
        #y_train, y_test = (
        #    to_categorical(y_train, 2).astype(np.float64),  #to_categorical 特徴ラベルをone-hotベクトルにする　クラスベクトルをバイナリクラス行列に変換
        #    to_categorical(y_test, 2).astype(np.float64),
        #)
        # Training
        #model = ExtremeLearningMachine(n_unit=n_unit, activation=None)      #ELM
        model = xgb.XGBClassifier(max_depth=3)      #XGBoost        ??
        #model.fit(x_train_std, y_train)     #固定回数の試行でモデルを学習させる
        model.fit(x_train,y_train)
        # Results
    
        y_true_train = y_train
        y_pred_train = model.predict(x_train)
        y_true_test = y_test
        y_pred_test = model.predict(x_test)
    
        # evaluate train 
        acc_train_total.append(accuracy_score(y_true_train, y_pred_train))      #結果を格納して追加していってる
        precision_train_total.append(precision_score(y_true_train, y_pred_train))
        recall_train_total.append(recall_score(y_true_train, y_pred_train))
        f1_train_total.append(f1_score(y_true_train, y_pred_train))
        # evaluate test     
        acc_test_total.append(accuracy_score(y_true_test, y_pred_test))
        precision_test_total.append(precision_score(y_true_test, y_pred_test))
        recall_test_total.append(recall_score(y_true_test, y_pred_test))
        f1_test_total.append(f1_score(y_true_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        # permutation importance
        #if threshold == 32 and pi_mode:     #変更
        #    perm_list = compute_permutaion_importance(      #l95の関数
        #        perm_list, x_train_std, y_train, model, columns_dic
        #    )


    eval_result["train_accuracy"] = np.average(acc_train_total)     #5つの平均計算
    eval_result["train_precision"] = np.average(precision_train_total)
    eval_result["train_recall"] = np.average(recall_train_total)
    eval_result["train_f1"] = np.average(f1_train_total)

    eval_result["test_accuracy"] = np.average(acc_test_total)
    eval_result["test_precision"] = np.average(precision_test_total)
    eval_result["test_recall"] = np.average(recall_test_total)
    eval_result["test_f1"] = np.average(f1_test_total)

    eval_df = pd.DataFrame.from_dict(eval_result, orient="index")       #from_dict:辞書型からDataFrameの生成
    #eval_df = eval_df.rename("testplay")        #行0をthresholdに変更

    # Output results
    eval_df.to_csv(
        "./output/eval_mode1.csv"
    )

if testmode==2:
    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
        print(f"Fold times:{n_fold}")
        x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
        y_train, y_test = y_data[trn_idx], y_data[val_idx]
        #x_train_std, x_test_std, _ = standard_trans(x_train, x_test)    #l26
        #y_train, y_test = (
        #    to_categorical(y_train, 2).astype(np.float64),  #to_categorical 特徴ラベルをone-hotベクトルにする　クラスベクトルをバイナリクラス行列に変換
        #    to_categorical(y_test, 2).astype(np.float64),
        #)
        # Training
        #model = ExtremeLearningMachine(n_unit=n_unit, activation=None)      #ELM
        #model = xgb.XGBClassifier(max_depth=3)      #XGBoost        ??
        #model.fit(x_train_std, y_train)     #固定回数の試行でモデルを学習させる
        #model.fit(x_train,y_train)

        # xgboostモデルの作成
        clf = xgb.XGBClassifier(verbosity= 0)       #verbosity??

        # ハイパーパラメータ探索
        clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
        clf_cv.fit(x_train, y_train)
        print("1")
        print(clf_cv.best_params_, clf_cv.best_score_)

        # 改めて最適パラメータで学習
        clf = xgb.XGBClassifier(**clf_cv.best_params_)
        clf.fit(x_train, y_train)

        # 学習モデルの保存、読み込み
        # import pickle
        # pickle.dump(clf, open("model.pkl", "wb"))
        # clf = pickle.load(open("model.pkl", "rb"))

        # 学習モデルの評価
        #pred = clf.predict(x_test)
        #print("2")
        #print(confusion_matrix(y_test, pred))
        #print("3")
        #print(classification_report(y_test, pred))

        y_true_train = y_train
        y_pred_train = clf.predict(x_train)
        y_true_test = y_test
        y_pred_test = clf.predict(x_test)
    
        # evaluate train 
        acc_train_total.append(accuracy_score(y_true_train, y_pred_train))      #結果を格納して追加していってる
        precision_train_total.append(precision_score(y_true_train, y_pred_train))
        recall_train_total.append(recall_score(y_true_train, y_pred_train))
        f1_train_total.append(f1_score(y_true_train, y_pred_train))
        # evaluate test     
        acc_test_total.append(accuracy_score(y_true_test, y_pred_test))
        precision_test_total.append(precision_score(y_true_test, y_pred_test))
        recall_test_total.append(recall_score(y_true_test, y_pred_test))
        f1_test_total.append(f1_score(y_true_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        # permutation importance
        #if threshold == 32 and pi_mode:     #変更
        #    perm_list = compute_permutaion_importance(      #l95の関数
        #        perm_list, x_train_std, y_train, model, columns_dic
        #    )


    eval_result["train_accuracy"] = np.average(acc_train_total)     #5つの平均計算
    eval_result["train_precision"] = np.average(precision_train_total)
    eval_result["train_recall"] = np.average(recall_train_total)
    eval_result["train_f1"] = np.average(f1_train_total)

    eval_result["test_accuracy"] = np.average(acc_test_total)
    eval_result["test_precision"] = np.average(precision_test_total)
    eval_result["test_recall"] = np.average(recall_test_total)
    eval_result["test_f1"] = np.average(f1_test_total)

    eval_df = pd.DataFrame.from_dict(eval_result, orient="index")       #from_dict:辞書型からDataFrameの生成
    #eval_df = eval_df.rename("testplay")        #行0をthresholdに変更

    # Output results
    eval_df.to_csv(
        "./output/eval_mode2.csv"
    )

if testmode==3:
    for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
        print(f"Fold times:{n_fold}")
        x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
        y_train, y_test = y_data[trn_idx], y_data[val_idx]

        # xgboostモデルの作成
        clf = xgb.XGBClassifier(verbosity= 0)       #verbosity??

        # ハイパーパラメータ探索
        clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
        clf_cv.fit(x_train, y_train)
        print("1")
        print(clf_cv.best_params_, clf_cv.best_score_)

        # 改めて最適パラメータで学習
        clf = xgb.XGBClassifier(**clf_cv.best_params_)
        clf.fit(x_train, y_train)

        fti = clf.feature_importances_  

        print('Feature Importances:')
        for i, feat in enumerate(features):
            print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

        y_true_train = y_train
        y_pred_train = clf.predict(x_train)
        y_true_test = y_test
        y_pred_test = clf.predict(x_test)
    
        # evaluate train 
        acc_train_total.append(accuracy_score(y_true_train, y_pred_train))      #結果を格納して追加していってる
        precision_train_total.append(precision_score(y_true_train, y_pred_train))
        recall_train_total.append(recall_score(y_true_train, y_pred_train))
        f1_train_total.append(f1_score(y_true_train, y_pred_train))
        # evaluate test     
        acc_test_total.append(accuracy_score(y_true_test, y_pred_test))
        precision_test_total.append(precision_score(y_true_test, y_pred_test))
        recall_test_total.append(recall_score(y_true_test, y_pred_test))
        f1_test_total.append(f1_score(y_true_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        # permutation importance
        #if threshold == 32 and pi_mode:     #変更
        #    perm_list = compute_permutaion_importance(      #l95の関数
        #        perm_list, x_train_std, y_train, model, columns_dic
        #    )


    eval_result["train_accuracy"] = np.average(acc_train_total)     #5つの平均計算
    eval_result["train_precision"] = np.average(precision_train_total)
    eval_result["train_recall"] = np.average(recall_train_total)
    eval_result["train_f1"] = np.average(f1_train_total)

    eval_result["test_accuracy"] = np.average(acc_test_total)
    eval_result["test_precision"] = np.average(precision_test_total)
    eval_result["test_recall"] = np.average(recall_test_total)
    eval_result["test_f1"] = np.average(f1_test_total)

    eval_df = pd.DataFrame.from_dict(eval_result, orient="index")       #from_dict:辞書型からDataFrameの生成
    #eval_df = eval_df.rename("testplay")        #行0をthresholdに変更

    # Output results
    eval_df.to_csv(
        "./output/eval_mode3.csv"
    )

#if threshold == 32 and pi_mode:     #変更
#output_perm_imp(perm_list, columns_dic, n_unit)     #l129の関数

#Training(x_data, y_data, n_unit, threshold, save_mode=True, shi_work=shi_work)
