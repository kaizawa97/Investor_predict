# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

path = "投資メモcsvファイル"

files = os.listdir(path)
  # ディレクトリにあるファイル一覧を取得

files.remove(".DS_Store")

i = []

for filename in files:
  fullpath = path + "/" + filename
 # ファイルを読み込みます 
  print(filename)
  df = pd.read_csv(fullpath, encoding="shift-jis", header=0)
  i.append(df)

countup = len(df) #要素数の指定するらしいです。

updata = []
print(df.head())
for i in range(1, countup):
    updata.append(float(df.loc[i, ['close']] - df.loc[i - 1, ['close']]) / float(df.loc[i - 1, ['close']]) * 20)  #株価の上昇率を算出するらしいです。

countm = len(updata)

successivedata = []  #4日分のデータを格納

answers = []  #正解の値を入れるデータ

for i in range(4, countm):  # 連続の上昇率のデータを格納していきます
    successivedata.append([updata[i-4], updata[i-3], updata[i-2], updata[i-1]])
    # 上昇率が0以上なら1、そうでないなら0を格納します
    if updata[i] > 0:
        answers.append(1)
    else:
        answers.append(0)

X_train, X_test, y_train, y_test = train_test_split(successivedata, answers, train_size=0.8, test_size=0.2, random_state=1)
  # データの分割（データの80%を訓練用に、20％をテスト用に分割する）

# clf = linear_model.SGDClassifier()  # 確率的勾配降下法
# clf.fit(clf, X_train, y_train, X_test, y_test)

# clf = tree.DecisionTreeClassifier()
# clf.fit(clf, X_train, y_train,X_test, y_test)  # 決定木

clf = svm.LinearSVC()  # サポートベクターマシーン
clf.fit(X_train, y_train)  # サポートベクターマシーンによる訓練

# 学習後のモデルによるテスト
y_train_pred = clf.predict(X_train)  # トレーニングデータを用いた予測
y_val_pred = clf.predict(X_test)  # テストデータを用いた予測

#パラメーターを最適化していきます
parameters = {'C': [1, 3, 5], 'loss': ('hinge', 'squared_hinge')}  # グリッドサーチするパラメータを設定

clf = GridSearchCV(svm.LinearSVC(), parameters)
clf.fit(X_train, y_train)  # グリッドサーチを実行
GS_C, GS_loss = clf.best_params_.values()  #グリットサーチを最適化します

print("最適パラメータ：{}".format(clf.best_params_))

# 最適パラメーターを指定して再度学習
clf = svm.LinearSVC(loss=GS_loss, C=GS_C)
clf.fit(X_train, y_train)

# 再学習後のモデルによるテスト
y_train_pred = clf.predict(X_train)  # トレーニングデータを用いた予測
y_val_pred = clf.predict(X_test)  # テストデータを用いた予測

# 正解率の計算
train_score = accuracy_score(y_train, y_train_pred)
test_score = accuracy_score(y_test, y_val_pred)

# 正解率を表示
print("トレーニングデータに対する正解率：" + str(train_score * 100) + "%")
print("テストデータに対する正解率：" + str(test_score * 100) + "%")
