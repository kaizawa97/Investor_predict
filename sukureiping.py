import urllib
import ssl
import os
import pandas as pd
import time
import random
import glob
import csv
from bs4 import BeautifulSoup
import time
import random

code = "4726"
year = "2003"
path = "投資メモcsvファイル/"
intyear = int(year)
print("入力待ちです。しばらくお待ち下さい！")
n = int(input())
for year in range(n):
    intyear += 1
    year = str(intyear)
    print(year)
    url = "https://kabuoji3.com/stock/"+code+"/"+year+"/"
    ssl._create_default_https_context = ssl._create_unverified_context
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, "html.parser")
    timerandom = random.uniform(10, 20)
    print(timerandom)
    time.sleep(timerandom)
    stockdata = soup.find_all("td")
    stockdata = [s.contents[0] for s in stockdata]
    stockdata = list(zip(*[iter(stockdata)]*7))
    df = pd.DataFrame(stockdata,)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'valume', 'adjclose']
    df = df.set_index("date")
    df.to_csv(path+year+code+".csv")
