特徴量のCSVファイル出力  
for_csv/thirdparty/geoip/GeoLite2-City.mmdb  
for_csv/features/doc2vec.model  
条件とするドメイン数による  
for_csv/blacklist_4000.txt  
for_csv/tranco_4000.txt  

example.pyの実行時  1ドメインの特徴量収集  
server/get_features/doc2vec_model/doc2vec.model  

ex_doc2vec.pyの実行時  
mode=1  悪性ドメインからJSデータをnpyに変換  
server/get_features/doc2vec_model/blacklist.txt  
mode=2  良性ドメインからJSデータをnpyに変換  
server/get_features/doc2vec_model/tranco_100k.txt  
mode=3  悪性・良性のJSデータからdoc2vec.model作成  
server/get_features/doc2vec_model/np_black_jsdata.npy  
server/get_features/doc2vec_model/np_white_jsdata.npy  