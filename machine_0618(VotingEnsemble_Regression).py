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

# 기본모델 적용
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

linear_model = LinearRegression()
rnf_model = RandomForestRegressor(random_state=42)

voting_regressor = VotingRegressor(estimators=[('lr', linear_model), ('rf',rnf_model)])
voting_regressor.fit(X_scaled_train, y_train)

pred_train = voting_regressor.predict(X_scaled_train)
print("훈련테스트 정확도 : ", voting_regressor.score(X_scaled_train, y_train))

pred_test = voting_regressor.predict(X_scaled_test)
print("테스트데이터 정확도 : ",voting_regressor.score(X_scaled_test, y_test))

# RMSE (Root Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE : ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE : ", np.sqrt(MSE_test))