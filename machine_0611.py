import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("file/Fvote.csv", encoding="utf-8")

# 특성(x)과 레이블(y) 나누기
X = data.loc[:, 'gender_female':'score_intention']
y = data[['vote']]

# train-test 데이터셋 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 연속형 특성의 Scaling
from sklearn.preprocessing import MinMaxScaler
scaler_minmax = MinMaxScaler()

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

# Standardization 스케일링
from sklearn.preprocessing import StandardScaler
scaler_standard = StandardScaler()

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)
X_scaled_standard_test = scaler_standard.transform(X_test)

# 모델 학습
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_scaled_minmax_train, y_train)
pred_train = model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)

# 혼동행렬
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
# print("훈련데이터 오차행렬:\n", confusion_train)

confusion_test = confusion_matrix(y_test, pred_test)
# print("테스트데이터 오차행렬:\n", confusion_test)

# Standardize 정규화 데이터 적용결과
model.fit(X_scaled_standard_train, y_train)
pred_train1 = model.predict(X_scaled_standard_train)
model.score(X_scaled_standard_train, y_train)

pred_test1 = model.predict(X_scaled_standard_test)
model.score(X_scaled_standard_test, y_test)

from sklearn.metrics import confusion_matrix
confusion_train1 = confusion_matrix(y_train, pred_train1)
print("훈련데이터:\n", confusion_train)

confusion_test1 = confusion_matrix(y_test, pred_test1)
print("테스트데이터:\n", confusion_test1)