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

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}


def get_html(url):
    request = urllib.request.Request('http://'+url, headers=headers)
    try:
        resp = urllib.request.urlopen(request, timeout=5)
        return resp.read()
    except (urllib.error.HTTPError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        print("ERROR:",url)
        return None
    except:
        raise

#htmlからjsのリンクをとってくる
def get_js_link(url):
    html = get_html(url)
    if html:
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
    else:
        return None, None

#リンクから文字列をとってくる
def link_to_code(url, link):
    if link!='':
        try:
            contents = urllib.request.urlopen("http://"+url+"/"+ link,timeout=5)
            return contents.read()
        except (urllib.error.HTTPError,http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
            UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
            print("ERROR:",link)
            return None
        except:
            raise
    else:
        print('None')
        return None

def url_to_code(url):
    try:
        contents = urllib.request.urlopen(url,timeout=5)
        js = contents.read()
        return js
    except (urllib.error.HTTPError,http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        print("ERROR:",url)
        return None
    except:
        raise

#decodeして１行ずつ分割
def decode_and_split(js):
    jssplit_result = []
    decodejs = js.decode('utf-8')
    jssplit = decodejs.split('\n')
    
    for a in jssplit:
        if a !='':
            a = a.replace(' ','')
            jssplit_result.append(a)

    return jssplit_result

f1 = open('test.txt','r',encoding='utf-8', errors="", newline="") 
#f2 = open('test.txt','r',encoding='utf-8', errors="", newline="")

whitelist = f1.readlines()
#blacklist = f2.readlines()
jsdata=[]
for data in whitelist: 
    if not data == '':
        data = data.rstrip('\n')
        domain = data.rstrip('\r')
        print(domain)
        jslink, jsurllink = get_js_link(domain)
        print(jslink,jsurllink)
        if jslink!=None:
            for index,value in enumerate(jslink):
                js = link_to_code(domain, value)
                if js:
                    jssplit = decode_and_split(js)
                    jsdata.append(jssplit)

        if jsurllink!=None:         
            for index,value in enumerate(jsurllink):
                js = url_to_code(value)
                if js:
                    jssplit = decode_and_split(js)
                    jsdata.append(jssplit)

#print(len(jsdata))

"""
for data in blacklist: 
    if not data == '':
        data = data.rstrip('\n')
        domain = data.rstrip('\r')
        print(domain)
        jslink, jsurllink = get_js_link(domain)
        print(jslink,jsurllink)
        if jslink!=None:
            for index,value in enumerate(jslink):
                js = link_to_code(domain, value)
                if js:
                    jssplit = decode_and_split(js)
                    jsdata.append(jssplit)

        if jsurllink!=None:         
            for index,value in enumerate(jsurllink):
                js = url_to_code(value)
                if js:
                    jssplit = decode_and_split(js)
                    jsdata.append(jssplit)
"""


trainings = [TaggedDocument(data, [i]) for i,data in enumerate(jsdata)]

# トレーニング
m = Doc2Vec(documents= trainings, vector_size=1, dm = 0, alpha=0.003, window=16, min_count=5, sample=0.001)
# iter=200

# モデルのセーブ
#m.save("doc2vec.model")
m.save("test.model")

f1.close()
#f2.close()