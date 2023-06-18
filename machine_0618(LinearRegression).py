### 회귀(Regression) ###

# 분석데이터 준비
import warnings
warnings.filterwarnings("ignore")

# 분석데이터 준비
import pandas as pd
data = pd.read_csv("file/housing_price.csv", encoding='utf-8')
X = data[data.columns[1:5]]
y = data[['house_value']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

# statmodel 적용
import statsmodels.api as sm
x_train_new = sm.add_constant(X_train)
x_test_new = sm.add_constant(X_test)

multi_model = sm.OLS(y_train, x_train_new).fit()
print(multi_model.summary())

multi_model2 = sm.OLS(y_test, x_test_new).fit()
print(multi_model2.summary())

# sckit-learn 적용
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
pred_test = model.predict(X_scaled_test)
print("훈련데이터 정확도 : ", model.score(X_scaled_train, y_train))
print("테스트데이터 정확도 : ", model.score(X_scaled_test, y_test))

# RMSE (Root Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련데이터 RMSE : ", np.sqrt(MSE_train))
print("테스트데이터 RMSE : ", np.sqrt(MSE_test))

# 기타 선형 모델평가지표 : MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, pred_test))

# 기타 선형 모델평가지표 : MSE (Mean Squared Error)
print(mean_squared_error(y_test, pred_test))

# 기타 선형 모델평가지표 : MAPE (Mean Absolute Percentage Error)
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - pred_test) / y_test)) *100
print(MAPE(y_test, pred_test))

# 기타 선형 모델평가지표 : MPE (Mean Percentage Error)
def MPE(y_test, y_pred):
    return np.mean((y_test - pred_test) / y_test) * 100
print(MPE(y_test, pred_test))