import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import mpl_finance as mpf
from matplotlib.dates import date2num

#pathを指定します
path = "投資メモcsvファイル"

files = os.listdir(path)
#フォルダから除外します
files.remove(".DS_Store") 

#配列を作成します
i = []
filename = [] 

for filename in files:
  fullpath = path + "/" + filename
 # ファイルを読み込みます
  df = pd.read_csv(fullpath, encoding="shift-jis", header=0)
  i.append(df)

  #csvのファイルによって"date"の部分は異なります
  df["date"] = pd.to_datetime(df["date"])
  #私のcsvファイルでは自動的にインデックスが入っていたのでdateの部分をindexにしました。
  df.set_index('date', inplace=True)

  df_w = df.copy()
  df_w.index = mdates.date2num(df_w.index)
  data_w = df_w.reset_index().values

  fig = plt.figure(figsize=(12, 4))
  ax = fig.add_subplot(1, 1, 1)

  #widthやalphaの部分は好きに変更してください
  mpf.candlestick_ohlc(ax, data_w, width=0.7, alpha=0.5,
                     colorup='g', colordown='r')
  #色も変更できます

  locator = mdates.AutoDateLocator()
  ax.xaxis.set_major_locator(locator)
  ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

  #保存してあげるのに毎回実行するのは面倒なのでfor分にしました。
  for n in range(1):
    n = str(n)
    y = str(filename) + n + ".png"
    print(y)
    plt.savefig('ローソク足画像ファイル入れ/' + y)

ax.grid()
ax.legend()
plt.show()
