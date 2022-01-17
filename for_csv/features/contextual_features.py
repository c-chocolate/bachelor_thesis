import http.client
from io import BytesIO
import os
import socket
import sys
import urllib.error
import urllib.request

from bs4 import BeautifulSoup
import tldextract as tld
import whois

def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}


def get_html(url):
    request = urllib.request.Request('http://'+url, headers=headers)
    try:
        resp = urllib.request.urlopen(request, timeout=5)
        return resp.read()
    except (http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        return None
#        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#        conn.connect((url, 80))
#        buffer = BytesIO()
#        while True:
#            chunk = conn.recv(4096)
#            if chunk:
#                buffer.write(chunk)
#            else:
#                break
#        return buf.getvalue().decode('utf-8')
    except:
        raise


def contextual_features(url):
    domain = url.rstrip('\n')
    ext = tld.extract(domain)
    domain1 = '.'.join(filter(None, [ext.domain, ext.suffix]))

    num_labels = 0
    life_time = 0
    active_time = 0

    html = None
    try:
        html = get_html(domain)
    except (urllib.error.HTTPError, urllib.error.URLError, 
        ConnectionResetError, socket.timeout):
        try:
            html = get_html(domain1)
        except (urllib.error.HTTPError, urllib.error.URLError, 
            ConnectionResetError, socket.timeout):
            pass

    if html:
        try:
            soup = BeautifulSoup(html, features='html.parser')
            num_labels = len(soup.find_all())
        except UnboundLocalError: # probably bug
            pass

    create = None
    update = None
    expire = None
    try:
        disable_print()
        w = whois.whois(domain1)
        enable_print()
        if w.creation_date:
            if isinstance(w.creation_date, list):
                create = w.creation_date[0]
            elif isinstance(w.creation_date, str):
                pass 
            else:
                create = w.creation_date
            if isinstance(create, str):
                create = None
        if w.updated_date:
            if isinstance(w.updated_date, list):
                update = w.updated_date[0]
            elif isinstance(w.updated_date, str):
                pass 
            else:
                update = w.updated_date
            if isinstance(update, str):
                update = None
        if w.expiration_date:
            if isinstance(w.expiration_date, list):
                expire = w.expiration_date[0]
            elif isinstance(w.expiration_date, str):
                pass 
            else:
                expire = w.expiration_date
            if isinstance(expire, str):
                expire = None
    except whois.parser.PywhoisError:
        pass

    if expire and create:
        td = expire - create
        life_time = td.days

    if update and create:
        td = update - create
        active_time = td.days
    elif life_time > 0:
        active_time = life_time

    return num_labels, life_time, active_time
        

if __name__ == '__main__':
    print(contextual_features('www-infosec.ist.osaka-u.ac.jp'))
