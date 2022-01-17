## Virtualenv
全てのスクリプトは virtualenv の利用を前提とし, 環境の名前は **venv** とします. src/ で,

    $ python3 -m venv venv
    $ source venv/bin/activate #環境をアクティベート
    $ pip install -U pip
    $ pip install wheel
    $ pip install -r requirements.txt

で環境設定完了.
環境が必要でなくなったら`$ deactivate`.

## リスト生成

venv 環境でやってください.

まず lists/ に移動し, get_lists.sh を実行する当日のリストを生成できます.
[Phishtank](https://phishtank.com) の API キーが必要なので, 取得して "api-key-here" のところに入れ替えてください.
process.py は自動で呼び出されるので手動で実行させる必要はない.

## 特徴量抽出

venv 環境でやってください.

src/ に戻ります.
抽出する前に並行ジョブ数を設定してください.
generate.py の 47 行目の `n_jobs=96` のところ, 96 を適当な数で入れ替えてください.
目安として, システムメモリ / 130 の値で大丈夫かと思います.
generate.sh (**generate.py ではない**) を実行させると抽出されます.
generate.sh と generate.py の中, パスの変更が必要の可能性もあるので, 適宜に変更してください.

抽出する特徴量は 0_日付.csv 1_日付.csv となります.
0 は benign で 1 は malicious.
また, 特徴量の順番は修論と若干異なります.
マッピングは **feature_mapping** を参照してください.

## 学習 & 交差検証
venv 環境でやってください.

learn.py を実行.
12 と 13 行目のファイル名を適宜に修正してください.
