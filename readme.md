-----特徴量のCSVファイル出力-----  
python generate.py [ファイル名] 2>&1 > [ファイル]名.log  
generate.sh Chengさんの実行コマンド  
testgene.py 追加特徴量なし  
testweb.py 追加特徴量の1つのドメインのみ  
randomtext.py ドメインリストをランダムに並び替える  
test.txt test2.txt  テスト用のドメインリスト  
--必要なファイル--  
for_csv/thirdparty/geoip/GeoLite2-City.mmdb  
for_csv/features/doc2vec.model  
条件とするドメイン数による  
for_csv/blacklist_4000.txt  
for_csv/tranco_4000.txt  

-----1ドメインの特徴量収集-----  
python example.py  
en_to_ja.txt    英和辞書
--必要なファイル--  
server/get_features/doc2vec_model/doc2vec.model  

-----Doc2Vecのモデル作成-----
python ex_doc2vec.py  
--必要なファイル--  
mode=1  悪性ドメインからJSデータをnpyに変換  
server/get_features/doc2vec_model/blacklist.txt  
mode=2  良性ドメインからJSデータをnpyに変換  
server/get_features/doc2vec_model/tranco_100k.txt  
mode=3  悪性・良性のJSデータからdoc2vec.model作成  
server/get_features/doc2vec_model/np_black_jsdata.npy  
server/get_features/doc2vec_model/np_white_jsdata.npy  

-----その他-----
traning_elm MADMAXより
src Chengさんのデータ