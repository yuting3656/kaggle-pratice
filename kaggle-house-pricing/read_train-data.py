import numpy as py
import pandas as pd
import csv

from sklearn.ensemble import GradientBoostingClassifier

# 讀取 train data set
df_train = pd.read_csv('train.csv')
# 讀取 test data set
df_test = pd.read_csv('test.csv')

# missing data
missing_total = df_train.isnull().sum().sort_values(ascending=False)
pd_missing_total = pd.concat([missing_total], axis=1, keys=['total'])
print(pd_missing_total)
# drop 空值
df_train = df_train.drop( (pd_missing_total[pd_missing_total['total'] > 1]).index , 1)

# 拿出 y: SalePrice 當 y_train
y_train = df_train['SalePrice']

# 刪除 y: SalePrice 做 x_train
x_train = df_train.drop('SalePrice', axis=1)

# 轉換 string to int
x_train = pd.get_dummies(x_train)


gb_clf = GradientBoostingClassifier(
    criterion='friedman_mse',
    n_estimators=2,
    learning_rate=0.1,
    max_depth=3,
    max_features="auto"
)

gb_clf.fit(x_train, y_train)

y_pred = gb_clf.predict(df_test)


with open ('final.csv', 'w', newline='') as csv_writer:
    hous_price_y_pred_r = csv.writer(csv_writer, delimiter=',', lineterminator='\n')
    hous_price_y_pred_r.writerow(y_pred)
