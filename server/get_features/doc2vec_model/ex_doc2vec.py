from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import http.client
from io import BytesIO
import os
import socket
import sys
import urllib.error
import urllib.request
import requests
import re
import copy
import csv
import pandas as pd
import numpy as np

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}


def get_html(url):
    request = urllib.request.Request('http://'+url, headers=headers)
    try:
        resp = urllib.request.urlopen(request, timeout=5)
        return resp.read()
    except (socket.timeout, urllib.error.HTTPError, urllib.error.URLError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError,ConnectionResetError) as e: # possibly plaintext or HTTP/1.0
        print("ERROR:",e,url)
        return None
    except:
        raise

#htmlからjsのリンクをとってくる
def get_js_link(url):
    html = get_html(url)
    if html:
        try:
            decodehtml = html.decode('utf-8')
            decodehtml = decodehtml.replace(' ','')     #空白を抜く
            #src= or href=で始まり.jsで終わる文字列検索
            findjssrc = re.findall(r'src=[\",\'][a-z,A-Z,0-9,\-,\_,\.,\!,\',\(,\),\~,\s,\/]+\.js[\",\']',decodehtml) 
            findjshref = re.findall(r'href=[\",\'][a-z,A-Z,0-9,\-,\_,\.,\!,\',\(,\),\~,\s,\/]+\.js[\",\']',decodehtml) 
            findjs = findjssrc + findjshref
            #print(findjs)   
            findjsurl = re.findall(r'https?://[a-z,A-Z,0-9,\-,\_,\.,\!,\',\(,\),\~,\s,\/]+\.js[\",\']',decodehtml)
            #print(findjsurl) 

            for index, value in enumerate(findjs):
                findjs[index] = value.replace('src=','')
                findjs[index] = findjs[index].replace('href=','')
                findjs[index] = findjs[index].replace('\"','')
                findjs[index] = findjs[index].replace("\'","")
                
                if findjs[index][:2]=='//':#頭が//だったらhttps:加えて、findjsurlに変更
                    findjsurl.append('https:' + findjs[index])
                    findjs[index] = ''

            for index, value in enumerate(findjsurl):
                findjsurl[index] = value.replace('\"','')
                findjsurl[index] = findjsurl[index].replace("\'","")
            return findjs, findjsurl
        except (UnicodeDecodeError):
            print("UnicodeDecodeError")
            return None, None
    else:
        return None, None

#リンクから文字列をとってくる
def link_to_code(url, link):
    if link!='':
        try:
            contents = urllib.request.urlopen("http://"+url+"/"+ link,timeout=5)
            return contents.read()
        except (socket.timeout, urllib.error.HTTPError, urllib.error.URLError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError,ConnectionResetError): # possibly plaintext or HTTP/1.0
            print("ERROR:",link)
            return None
        except:
            raise
    else:
        #print('None')
        return None

def url_to_code(url):
    try:
        contents = urllib.request.urlopen(url,timeout=5)
        js = contents.read()
        return js
    except (socket.timeout, urllib.error.HTTPError, urllib.error.URLError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError,ConnectionResetError): # possibly plaintext or HTTP/1.0
        print("ERROR:",url)
        return None
    except:
        raise

#decodeして１行ずつ分割
def decode_and_split(js):
    jssplit_result = []
    try:
        decodejs = js.decode('utf-8')
        #jssplit = decodejs.split('\n')
        jssplit = re.split('[\n,\r,\r\n]', decodejs)

        for a in jssplit:
            if a !='':
                a = a.replace(' ','')
                jssplit_result.append(a)
    except (UnicodeDecodeError):
        print("UnicodeDecodeError")

    return jssplit_result

mode = 3 #0:test 1:black 2:white 3:model       !!

if mode == 0 :
    f1 = open('test.txt','r',encoding='utf-8', errors="", newline="") 
elif mode == 1:
    f1 = open('blacklist.txt','r',encoding='utf-8', errors="", newline="")
elif mode ==2:
    f2 = open('tranco_100k.txt','r',encoding='utf-8', errors="", newline="")

#blacklist:16637行

#domain_num = 0

if mode==0 or mode==1:
    blacklist = f1.readlines()
elif mode ==2:
    whitelist = f2.readlines()

#f3 = open('jsdatalist.csv', 'w')
#f4 = open('jsdata_domainnum.txt', 'w')
#for x in list_row:
#    f.write(str(x) + "\n")
#f.close()

#jsdata=[]
datanum=0

"""
if mode==2:
    f5 = open("./testcsv.csv","r")
    reader=csv.reader(f5)
    jsdata=[e for e in reader]
    #for x in f5:
    #    jsdata.append(x)
        #以下のようにしてしまうと、改行コードがlistに入ってしまうため注意
        #list_row.append(x)
    #jsdata = pd.read_csv("jsdatalist.csv").values.tolist()
    f5.close()
    modetwo_datanum=0       # jsdata_domainnum.txtの数-1入れる

    #print(jsdata[1][0])
 """   


index = 0

readmode=0 #0:なし 1:途中読み込み !!

if readmode==0:
    jsdata=[]
    modetwo_datanum=0
elif readmode==1:
    modetwo_datanum=1640     # !!
    get_data=np.load('np_white_jsdata.npy', allow_pickle=True)  #!!
    jsdata=get_data.tolist()
    print("READ FINISH")

if mode==0 or mode==1:
    for data in blacklist: 
        if not data == '' and datanum>=modetwo_datanum:
            data = data.rstrip('\n')
            domain = data.rstrip('\r')
            datanum+=1
            print("<<blacklist>>",datanum,domain)
            if get_html(domain):
                #domain_num += 1
                jslink, jsurllink = get_js_link(domain)
                print(jslink,jsurllink)
                if jslink!=None:
                    for index,value in enumerate(jslink):
                        js = link_to_code(domain, value)
                        if js:
                            jssplit = decode_and_split(js)
                            jsdata.append(jssplit)
                            np_jsdata= np.array(jsdata)
                            np.save('np_black_jsdata',np_jsdata)
                            #print(jssplit[0])
                            #f3.write(str(jssplit) + "\n")

                if jsurllink!=None:         
                    for index,value in enumerate(jsurllink):
                        js = url_to_code(value)
                        if js:
                            jssplit = decode_and_split(js)
                            jsdata.append(jssplit)
                            np_jsdata= np.array(jsdata)
                            np.save('np_black_jsdata',np_jsdata)
                            #f3.write(str(jssplit) + "\n")
            f4 = open('jsdata_domainnum.txt', 'w')
            f4.write(str(datanum)+str(domain)+"\n")
            f4.close()
        elif datanum<modetwo_datanum:
            datanum+=1
            data = data.rstrip('\n')
            domain = data.rstrip('\r')
            print("<<SKIP>>",datanum,domain)

#print(jsdata[0][0])

print(datanum)
#domainmax=datanum   #domainmax:blacklistで読み込んだ数

#redata=np.load('np_jsdata.npy', allow_pickle=True)
#reredata=redata.tolist()
#print(reredata[0][0])
#print("QQQ",jsdata==reredata)

index_white = 0   #whitelistの現在の数
if mode ==2 :
    domainmax=15521     # !!
    
    for data in whitelist: 
        if index_white >= domainmax:
            break
        if not data == '' and index_white>=modetwo_datanum:
            index_white +=1
            data = data.rstrip('\n')
            domain = data.rstrip('\r')
            #datanum+=1
            print("<<whitelist>>",index_white,domain)
            if get_html(domain):
                jslink, jsurllink = get_js_link(domain)
                print(jslink,jsurllink)
                if jslink!=None:
                    for index,value in enumerate(jslink):
                        js = link_to_code(domain, value)
                        if js:
                            jssplit = decode_and_split(js)
                            jsdata.append(jssplit)
                            np_jsdata= np.array(jsdata)
                            np.save('np_white_jsdata',np_jsdata)
                            #f3.write(str(jssplit) + "\n")

                if jsurllink!=None:         
                    for index,value in enumerate(jsurllink):
                        js = url_to_code(value)
                        if js:
                            jssplit = decode_and_split(js)
                            jsdata.append(jssplit)
                            np_jsdata= np.array(jsdata)
                            np.save('np_white_jsdata',np_jsdata)
                            #f3.write(str(jssplit) + "\n")
            f4 = open('jsdata_domainnum.txt', 'w')
            f4.write(str(index_white)+str(domain)+"\n")
            f4.close()
        elif index_white<modetwo_datanum:
            index_white+=1
            data = data.rstrip('\n')
            domain = data.rstrip('\r')
            print("<<SKIP>>",index_white,domain)


#f3.close()
#f4.close()

if mode==3:
    print("LOADING")
    get_black_data=np.load('np_black_jsdata.npy', allow_pickle=True)
    li_black_data=get_black_data.tolist()
    print("SIZE")
    print(len(li_black_data))
    get_white_data=np.load('np_white_jsdata.npy', allow_pickle=True)
    li_white_data=get_white_data.tolist()
    print(len(li_white_data))
    resize_white=li_white_data[:11637]
    jsdata=li_black_data+resize_white
#print(reredata[0][0])
#print("QQQ",jsdata==reredata)
#print("SIZE")
#print(len(li_black_data))
print(len(resize_white))
#print([len(v) for v in li_white_data])


if mode==0 or mode==3:
    print("TRAINING")
    trainings = [TaggedDocument(data, [i]) for i,data in enumerate(jsdata)]

    # トレーニング
    m = Doc2Vec(documents= trainings, vector_size=1, dm = 0, alpha=0.003, window=16, min_count=5, sample=0.001)
    # iter=200

    # モデルのセーブ
    if mode == 0:
        m.save("test.model")
    elif mode ==3:
        m.save("doc2vec.model")

if mode==0 or mode==1:
    f1.close()
elif mode == 2:
    f2.close()