# 분석 데이터 검토
import pandas as pd
data= pd.read_csv('file/house_price.csv', encoding='utf-8')

# 특성(x)과 레이블(y) 나누기
# 방법1 : 특성 이름으로 데이터셋(X) 나누기
X1 = data[['housing_age', 'income', 'bedrooms', 'households', 'rooms']]

# 방법2 : 특성 위치값으로 특성 데이터셋(X) 나누기
X2 = data[data.columns[0:5]]

# 방법3 : loc 함수로 특성 데이터셋(X) 나누기 (단, 불러올 특성이 연달아 있어야 함)
X3 = data.loc[:, 'housing_age':'rooms']

y = data[['house_value']]

# train-test 데이터셋 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state = 42)

# 정규화
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# train data의 정규화
scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

# test data의 정규화
X_scaled_minmax_test = scaler_minmax.transform(X_test)
X_scaled_standard_test = scaler_standard.transform(X_test)

# 모델 학습
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_minmax_train, y_train)

# 정확도(R-square : 설명력) (맞는 정도)
pred_train = model.predict(X_scaled_minmax_train)
model.score(X_scaled_minmax_train, y_train)

pred_test = model.predict(X_scaled_minmax_test)
model.score(X_scaled_minmax_test, y_test)

# RMSE (Root Mean Squared Error) (틀린 정도, 즉 오차를 판단)
import numpy as np
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, pred_test)
np.sqrt(MSE)

# MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, pred_test)

# MSE (Mean Squared Error)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred_test)

# MAPE (Mean Absolute Percentage Error)
def MAPE(y_test, pred_test) :
    return np.mean(np.abs((y_test - pred_test) / y_test)) * 100
MAPE(y_test, pred_test)

# MPE (Mean Percentage Error)
def MPE(y_test, pred_test):
    return np.mean((y_test-pred_test) / y_test) * 100
MPE(y_test, pred_test)

# 예측값 병합 및 저장
y_train['y_pred'] = pred_train
y_test['y_pred'] = pred_test

Total_test = pd.concat([X_test, y_test], axis=1)

Total_test.to_csv('file/regression_test.csv')