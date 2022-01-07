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

from bs4 import BeautifulSoup
import tldextract as tld
import whois

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}


def get_html(url):
    request = urllib.request.Request('http://'+url, headers=headers)
    try:
        resp = urllib.request.urlopen(request, timeout=5)
        return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        print("ERROR:", url)
        return None
    except:
        raise

#htmlからcssへのリンクを探す
def get_css_link(url):
    html = get_html(url)
    if html:
        decodehtml = html.decode('utf-8')

        #href=で始まり.cssで終わる文字列検索
        decodehtml = decodehtml.replace(' ','')
        findcss = re.findall(r'href=.+\.css',decodehtml)     
        for index, value in enumerate(findcss):
            findcss[index] = value.replace('href="','')     
        

        return findcss
    else:
        return None

#cssからcssへのリンクを探す
def get_loop_css(url, csslink, index):
    
    css = link_to_code(url, csslink[index])     #開く

    if css:
        decodecss = css.decode('utf-8')
        decodecss = decodecss.replace(' ','')

        #見ているcssのリンクをとってくる
        linkheader = re.search(r'.+\/',csslink[index]).group()

        loopcss = re.findall(r'url\(.+\.css', decodecss)     

        for index, value in enumerate(loopcss):     #いらないところを消す
            loopcss[index] = value.replace('url("', linkheader) 
        
            if not loopcss[index] in csslink:     #配列csslinkのなにかと一致しなかったら
                csslink.append(loopcss[index])
                csslink = get_loop_css(url, csslink, len(csslink)-1)

    return csslink


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
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
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
    except (urllib.error.HTTPError,http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        print("ERROR:",url)
        return None
    except:
        raise

#単語が存在するかどうか（複数形s es）まで
def judge_word(word, dictionaryword):
    judgeresult=0
    if word in dictionaryword:
        judgeresult=2
    elif word[-2:] == 'es':        #複数形es
        if word[:-2] in dictionaryword:
            judgeresult=3
    elif word[-1:] == 's':       #複数形s   
        if word[:-1] in dictionaryword:
            judgeresult=5
    return judgeresult


#分割して存在するか（2分割）
def split_two_judge(word,dictionaryword):
    result = 0
    for n in range(len(word)-3):  #前後二文字まで分割
        if judge_word(word[:n+2].lower(),dictionaryword)>0 and judge_word(word[n+2:].lower(),dictionaryword)>0:
            result += 7
            break
    return result

#分割して存在するか（3分割）
def split_three_judge(word,dictionaryword):
    result = 0
    for n in range(len(word)-3):  #前後二文字まで分割
        if judge_word(word[:n+2].lower(),dictionaryword)>0 and split_two_judge(word[n+2:],dictionaryword)>0:
            result += 11
            break
    return result

#単語判定
def judge_word_all(word,dictionaryword):
    result = 0
    result = judge_word(word, dictionaryword)    #そのまま or 複数形s,es
    if result == 0:
        result = split_two_judge(word,dictionaryword)   #2単語
        if result == 0:
            result = split_three_judge(word,dictionaryword) #3単語
    return result

#ベクトル
def get_js_comparison(domain):
        
    jslink, jsurllink= get_js_link(domain)
    #print(jslink,jsurllink)
    similar_list = []

    if jslink!=None:
        for index,value in enumerate(jslink):
            js = link_to_code(domain, value)

            if js:
                decodejs = js.decode('utf-8')
                jssplit = decodejs.split(' ')

                model = Doc2Vec.load('doc2vec_model/test.model')
                similar = model.infer_vector(jssplit)
                #print(similar)
                similar_list.append(similar[0])
        
    if jsurllink!=None:
        for index,value in enumerate(jsurllink):
            js = url_to_code(value)

            if js:
                decodejs = js.decode('utf-8')
                jssplit = decodejs.split(' ')
                
                model = Doc2Vec.load('doc2vec_model/test.model')
                similar = model.infer_vector(jssplit)
                #print(similar)
                similar_list.append(similar[0])
    
    if similar_list:
        return similar_list
    else:
        return None

def get_dictionary():

    dictionaryfile = open('en_to_ja.txt','r',encoding='UTF-8')
    dictionaryword = []
    for data in dictionaryfile:
        newdata = re.sub(r"[^a-zA-Z0-9]","",data.split()[0])
        #if not newdata in dictionaryword:
            #dictionaryword.append(newdata.lower())
        dictionaryword.append(newdata.lower())
    dictionaryfile.close()
    #print(len(dictionaryword))
    #削除後40275 40773
    #削除後（特殊文字除く）40522 
    #削除前46754
    return dictionaryword

def get_html_id_class(url):

    html = get_html(url)
    if html:
        decodehtml = html.decode('utf-8')

        #id名を探す
        decodehtml = decodehtml.replace(' ','')     #空白を抜く
        findidname = re.findall(r'id="[^"]+"', decodehtml)    
        for index, value in enumerate(findidname):
            findidname[index] = value.replace('id=','')
            findidname[index] = findidname[index].replace('\"','')
        findidname = list(set(findidname))        

        #class名を探す
        findclassname = re.findall(r'class="[^"]+"', decodehtml)    
        for index, value in enumerate(findclassname):
            findclassname[index] = value.replace('class=','')
            findclassname[index] = findclassname[index].replace('\"','')
        findclassname = list(set(findclassname))

        #id名とclass名のリストを結合
        findname = findidname + findclassname
        findname.sort()
        #findname.append('a-b-x_aw93az:sd.se9:')
        #print(findname)     #idとclass名のリスト
    else:
        findname=[]

    return findname

def dictionary_check(dictionaryword, findname):

    #存在するか判定
    resultnum = 0
    
    
    for s_data in findname:
        judgeresult = 1
        l=re.split('[\_,\-,0-9,\:,\.]',s_data)
        #print(l)
        for splitword in l:
            judgeresult*=judge_word_all(splitword.lower(),dictionaryword)
            
        #print(judgeresult, s_data)      #2:そのまま 3:es 5:s 7:2分割 11:3分割

        if judgeresult > 0:
            resultnum += 1

    return resultnum

class Add_Web_Features_Extractor:
    def __init__(self, domain):
        self.domain = domain.rstrip('\n')
        ext = tld.extract(self.domain)
        self.compact_domain = '.'.join(filter(None, [ext.domain, ext.suffix]))

        self.similar_list = get_js_comparison(self.domain)

        """
        self.__whois = False
        self.create = None
        self.update = None
        self.expire = None
        """

    def get_n_css_selectors(self):
       
       #htmlからcssのリンクを探す
        csslinkfromhtml = get_css_link(self.domain)   #csslink：配列でリンクが書かれている
        resultsum=0

        if csslinkfromhtml:
            csslink=copy.copy(csslinkfromhtml)

            #cssからcssへのリンクを探す
            for index, value in enumerate(csslinkfromhtml):
                csslink = get_loop_css(self.domain, csslink, index)

            #cssを開いて波括弧を数える
            
            for index,value in enumerate(csslink):
                css = link_to_code(self.domain, csslink[index])

                if css:
                    decodecss = css.decode('utf-8')
                    
                    cssresult = decodecss.count('{')
                    resultsum += cssresult

        return resultsum

    def get_html_id_class_num(self):
   
        dictionaryword = get_dictionary()

        findname = get_html_id_class(self.domain)

        #アンダーバーとかハイフンの記号に対処 classは日本語？

        resultnum = dictionary_check(dictionaryword, findname)
        
        return resultnum

    def get_html_id_class_rate(self):

        dictionaryword = get_dictionary()
        findname = get_html_id_class(self.domain)
        resultnum = dictionary_check(dictionaryword, findname)

        #print('resultnum',resultnum)
        #print('findname',len(findname))

        rate = 100*resultnum / len(findname)
        #%でいい？

        return rate

    def get_n_js_function(self):
        
        jslink, jsurllink = get_js_link(self.domain)   #jslink：配列でリンクが書かれている
        #print(jslink)
        #print(jsurllink)
        resultsum = 0
        
        if jslink!=None and jsurllink!=None:
            #print(jslink,jsurllink)
            for index, value in enumerate(jslink):
                jsresult = 0
                js = link_to_code(self.domain, value)
                #print(js)

                if js:
                    decodejs = js.decode('utf-8')
                    jsresult = decodejs.count('function')
                    resultsum += jsresult

            for index, value in enumerate(jsurllink):   #URLのjsのとき
                jsresult = 0
                js = url_to_code(value)

                if js:
                    decodejs = js.decode('utf-8')
                    jsresult = decodejs.count('function')
                    #print(value,jsresult)
                    resultsum += jsresult
                
        return resultsum

    def get_js_comparison_average(self):    #ベクトルの平均

        similar_list=self.similar_list
        if similar_list:
            similar_ave = sum(similar_list) / len(similar_list)
            return similar_ave
        
        else:
            return 0

    def get_js_comparison_max(self):    #ベクトルの最大

        similar_list=self.similar_list
        if similar_list:
            return max(similar_list)
        
        else:
            return 0

    def get_js_comparison_min(self):    #ベクトルの最小

        similar_list=self.similar_list
        if similar_list:
            return min(similar_list)
        
        else:
            return 0