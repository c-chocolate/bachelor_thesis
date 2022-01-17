import numpy as np      # numpyをnpという名前でインポート　numpy:数値計算ライブラリ
import pandas as pd     # pandas:データ分析
from sklearn.inspection import permutation_importance       # sklearn:機械学習ライブラリ    特徴評価のための置換重要度
from sklearn.model_selection import KFold       # データをトレーニングセットとテストセットに分割するためのトレーニング/テストインデックスを提供
from art.utils import to_categorical        # ラベルの配列をバイナリクラスマトリックスに変換
from sklearn.preprocessing import StandardScaler        # 平均値を取り除き、単位分散に合わせてスケーリングすることで、特徴を標準化
import math     # 数学関数
import sys      # 変数　関数
import argparse     # コマンドライン引数
import os       # os依存のモジュール
from elm_model import ExtremeLearningMachine        # ELM
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score     # 特定の目的のために予測誤差を評価する関数を実装

# プログラムの引数の対応
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", default=10, type=int)      # -t 特徴量の数
parser.add_argument("-n", "--n_unit", default=600, type=int)        # -n 中間ノード数
parser.add_argument("-f", "--file_name_path", default="file_name_path", type=str)       # -f ?? 出力ファイル名
args = parser.parse_args()

threshold = args.threshold          # -t
n_unit = args.n_unit                # -n
file_name = args.file_name_path     # -f


def standard_trans(x_train, x_test):        #l243
    stdsc = StandardScaler()        # データセットの標準化　特徴量の比率を揃える
    x_train_std = stdsc.fit_transform(x_train)      # 引数1つ目(x_train)を正規化　x_trainを小さくしたのがx_train_std
    # print(stdsc.mean_,stdsc.var_)
    x_test_std = stdsc.transform(x_test)        # 引数2つ目(x_test)を正規化
    return (
        x_train_std.astype(np.float64),     # astype 変換(float64bit)
        x_test_std.astype(np.float64),
        (stdsc.mean_, stdsc.var_),      # mean:平均　var:分散
    )


def shuffle_data(x_data, y_data):       #使ってない??
    n_data = np.shape(y_data)[0]        # 形状(各次元のサイズ)を取得
    shuffled_indices = np.arange(n_data)        # 間隔(公差)を指定
    np.random.shuffle(shuffled_indices)     # random.shuffle:引数をランダムに　??shuffled_indices
    x_data = x_data[shuffled_indices]       # ??x_dataはint?配列?
    y_data = y_data[shuffled_indices]       # ??
    return x_data, y_data


def extract_data_balanced(df):      #l190
    benign = df["label"] == 0       # benign:良性  benign:boolean型  csvのlabel列が0だったらTrue
    benign[24127:] = False      # 24127から最後まで　24127:収集したドメインの数
    df = pd.concat([df.loc[benign, :], df.loc[df["label"] == 1, :]], axis=0)        # concat:連結　axis=0:列に沿った処理　df.loc:配列から値取ってくるみたいな処理
    return df    #24127ずつ良性と悪性を取ってくる


def extract_data_unbalanced_more_malicious(df):        #l188
    # b:6050 m:19500     
    benign = df["label"] == 0       
    benign[6051:] = False       # ??6051は
    malicious = df["label"] == 1        
    malicious_index = np.where(malicious == True)       # where:条件を満たす要素のインデックスを取得    malicious_index:np.array型boolean
    malicious_index = malicious_index[0][:19500]        # 19500~を消す
    ind = np.zeros(len(malicious), dtype=bool)      # zeros:全ての要素を0(FALSE)とする配列を生成　maliciousと同じ長さのFALSE配列を作成
    ind[malicious_index] = True     # 要素malicious_indexだけTRUEに
    df = pd.concat([df.loc[benign, :], df.loc[ind, :]], axis=0)     # concat:連結　axis=0:列に沿った処理
    return df


def extract_data_unbalanced_more_benign(df):        #l186
    # b:19500 m:6050
    benign = df["label"] == 0       # extract_data_balancedと同じ??
    benign[19500:] = False      # extract_data_balancedは24127 ??19500は
    print(df.loc[benign, :].shape)      
    malicious = df["label"] == 1        
    malicious_index = np.where(malicious == True)       
    malicious_index = malicious_index[0][:6050]         
    print(len(malicious_index))         
    ind = np.zeros(len(malicious), dtype=bool)       
    ind[malicious_index] = True    
    df = pd.concat([df.loc[benign, :], df.loc[ind, :]], axis=0)    
    return df


def get_average_result(perm_list):      #l130
    result = pd.DataFrame(      #二次元の表形式データ
        data=0,     #DataFrameに格納するデータを配列などで指定
        columns=["importances_mean", "importances_std"],        #列名
        index=list(columns_dic.keys()),         #行名
    )
    for pi in perm_list:        #result表形式データにperm_list(引数)を入れていく
        result += pi
    result = result / len(perm_list)        #??len
    print(result)
    return result


def compute_permutaion_importance(perm_list, x_train_std, y_train, model, columns_dic):     #l267
    y_train_one_hot = np.argmax(y_train, axis=1)
    result = permutation_importance(
        model,
        x_train_std,
        y_train_one_hot,
        scoring="accuracy",     #acuracy
        n_repeats=10,       #機能を順列化する回数。
        n_jobs=-1,      #並行して実行するジョブ数
        random_state=71,        #各機能の順列を制御するための疑似乱数発生器
    )
    print(result)
    perm_imp_df = pd.DataFrame(     #二次元の表形式データ
        {
            "importances_mean": result["importances_mean"],     #平均
            "importances_std": result["importances_std"],   #標準偏差
        },
        index=list(columns_dic.keys()),     #特徴量が行
    )
    perm_list.append(perm_imp_df)       #リストに要素を追加
    return perm_list


def feature_selection(file_name, threshold, columns):       #l211
    if os.path.isfile(file_name):       #os.path.isfile:パスが存在しているファイルかどうか
        perm_imp_df = pd.read_csv(file_name, index_col=0)
        return list(perm_imp_df.index)[:threshold]
    else:
        if threshold == 25:         #??threshold 引数
            return columns          #??columns　引数
        else:
            raise ValueError("file not exists")


def output_perm_imp(perm_list, columns_dic, n_unit):        #l294
    result = get_average_result(perm_list)      #l82の関数
    perm_imp_df = pd.DataFrame(
        {
            "importances_mean": result["importances_mean"],     #??列名をどうするのか　l85のところ l109と同じ
            "importances_std": result["importances_std"],
        },
        index=list(columns_dic.keys()),
    )
    print(perm_imp_df)
    perm_imp_df = perm_imp_df.sort_values("importances_mean", ascending=False)      #accendingFalse:降順
    perm_imp_df.to_csv("./data/perm_imp_nunit{}.csv".format(n_unit))        #csv出力ファイル名、場所指定


def Training(x_data, y_data, n_unit, threshold, save_mode, shi_work):
    """
    Training a model by the entire dataset and save the trained model and the parameters.       #データセット全体でモデルを学習し、学習したモデルとそのパラメータを保存します。
    """
    stdsc = StandardScaler()        #importしたもの
    x_data = stdsc.fit_transform(x_data)         # 引数1つ目(x_data)を正規化
    y_data = to_categorical(y_data, 2).astype(np.float64)       #to_categorical 特徴ラベルをone-hotベクトルにする　クラスベクトルをバイナリクラス行列に変換　　　
    model = ExtremeLearningMachine(n_unit=n_unit)       #ELM
    model.fit(X=x_data, y=y_data)       #固定回数の試行でモデルを学習させる
    if not shi_work and save_mode:
        model.save_weights("./data/elm_threshold{}_nunit{}".format(threshold, n_unit))      #モデルを保存するときに使う
        mean, var = stdsc.mean_, stdsc.var_
        np.savez(       #複数のndarrayをnpzで保存
            "./data/param_threshold{}_nunit{}".format(threshold, n_unit),
            mean=mean,
            var=var,
        )
    elif shi_work and save_mode:
        model.save_weights("./data/elm_shi{}".format(n_unit))
        mean, var = stdsc.mean_, stdsc.var_
        np.savez(
            "./data/param_shi",
            mean=mean,
            var=var,
        )


# Define parameters
extraction_dataset_mode = (
    "normal"  # bengin than malicous("btm") or malicous than bengin ("mtb") or "normal"
)
shi_work = False
save_mode = False
pi_mode = True

####################################

# load data
df = pd.read_csv(file_name)     #csvファイルを読み込んでデータフレーム化
# Replace nan
df = df.replace(np.nan, 0)      #欠損値(nan)を0に置き換える
# Select extraction dataset mode
if extraction_dataset_mode == "btm":
    df = extract_data_unbalanced_more_benign(df)        #l67
elif extraction_dataset_mode == "mtb":
    df = extract_data_unbalanced_more_malicious(df)     #l54
elif extraction_dataset_mode == "normal":
    df = extract_data_balanced(df)      #l47

columns_dic = {
    column: index for index, column in enumerate(df.drop("label", axis=1).columns)      #df.drop:行を削除   axis1:行　.colums列を取ってくる　   特徴量の名前とインデックスのdict
}


if shi_work:
    features = [        #9つ??
        "length",
        "max_consecutive_chars",
        "entropy",
        "n_ip",
        "n_countries",
        "mean_TTL",
        "stdev_TTL",
        "life_time",
        "active_time",
    ]
else:
    #ここで特徴量を決定。
    features = feature_selection(       #l118
        "./data/perm_imp_nunit{}.csv".format(n_unit),   #n_unit:中間ノード数
        threshold=threshold,        #閾値
        columns=list(columns_dic.keys()),   #.keys:dictの値を取得   colums_dicのcolumnを取ってくる
    )

df = df.loc[:, features + ["label"]]        #df.loc:pandasの要素抽出(下限から)      featuresで選んだ特徴量とlabelからなるdfに上書き
x_data = df.drop("label", axis=1).values        #valuesでdfをndarray化　label以外
y_data = df["label"].values #labelだけ


FOLD_NUM = 5        #KFoldの分割数
fold_seed = 71      #乱数のシード指定
folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)      #Kfold:K-分割交差検証(データをn_split個に分け、n個を訓練用、k-n個をテスト用として使う)
fold_iter = folds.split(x_data)     #x_dataを区切り文字で分割   5個に分割
perm_list = []
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
for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):     #enumerate:for文でインデックス(n_fold)も利用できる      for5回
    print(f"Fold times:{n_fold}")
    x_train, x_test = x_data[trn_idx], x_data[val_idx]      #fold_iterから取ってくる
    y_train, y_test = y_data[trn_idx], y_data[val_idx]
    x_train_std, x_test_std, _ = standard_trans(x_train, x_test)    #l26
    y_train, y_test = (
        to_categorical(y_train, 2).astype(np.float64),  #to_categorical 特徴ラベルをone-hotベクトルにする　クラスベクトルをバイナリクラス行列に変換
        to_categorical(y_test, 2).astype(np.float64),
    )
    # Training
    model = ExtremeLearningMachine(n_unit=n_unit, activation=None)      #ELM
    model.fit(x_train_std, y_train)     #固定回数の試行でモデルを学習させる
    # Results
    y_train_pred = np.argmax(model.transform(x_train_std), axis=1)      #argmax:確率が高い方の要素番号を返す
    y_train_true = np.argmax(y_train, axis=1)       #元に戻している
    y_test_pred = np.argmax(model.transform(x_test_std), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    # evaluate train    
    acc_train_total.append(accuracy_score(y_train_true, y_train_pred))      #結果を格納して追加していってる
    precision_train_total.append(precision_score(y_train_true, y_train_pred))
    recall_train_total.append(recall_score(y_train_true, y_train_pred))
    f1_train_total.append(f1_score(y_train_true, y_train_pred))
    # evaluate test     
    acc_test_total.append(accuracy_score(y_test_true, y_test_pred))
    precision_test_total.append(precision_score(y_test_true, y_test_pred))
    recall_test_total.append(recall_score(y_test_true, y_test_pred))
    f1_test_total.append(f1_score(y_test_true, y_test_pred))
    # permutation importance
    if threshold == 25 and pi_mode:
        perm_list = compute_permutaion_importance(      #l95の関数
            perm_list, x_train_std, y_train, model, columns_dic
        )

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
if shi_work and save_mode:
    eval_df.to_csv("./output/eval_shi_nunit{}.csv".format(n_unit))      #formatで{}に代入
elif save_mode:
    eval_df.to_csv(
        "./output/eval_threshold{}_nunit{}.csv".format(
            threshold, n_unit
        )
    )
if threshold == 25 and pi_mode:
    output_perm_imp(perm_list, columns_dic, n_unit)     #l129の関数

# Training(x_data, y_data, n_unit, threshold, save_mode=True, shi_work=shi_work)
