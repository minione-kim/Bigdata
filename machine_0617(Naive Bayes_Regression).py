### 회귀(Regression) ###

# 분석데이터 준비
import warnings
warnings.filterwarnings("ignore")

# 분석데이터 준비
import pandas as pd
data = pd.read_csv("file/house_price.csv", encoding='utf-8')
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
from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
pred_test = model.predict(X_scaled_test)
print("훈련데이터 정확도 : ", model.score(X_scaled_train, y_train))
print("훈련데이터 정확도 : ", model.score(X_scaled_test, y_test))

# RMSE(Root Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련 데이터 RMSE : ", np.sqrt(MSE_train))
print("테스트 데이터 RMSE : ", np.sqrt(MSE_test))

param_grid = {'alpha_1' : [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3, 4],
              'lambda_1' : [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3, 4]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(grid_search.best_params_))
print("Best Score : {:.4f}".format(grid_search.best_score_))
print("TestSet Score : {:.4f}".format(grid_search.score(X_scaled_test, y_test)))

from scipy.stats import randint
param_distribs = {'alpha_1' : randint(low=1e-06, high=10),
                  'lambda_1' : randint(low=1e-06, high=10)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(BayesianRidge(), param_distributions=param_distribs, n_iter=50, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(random_search.best_params_))
print("Best Score : {:.4f}".format(random_search.best_score_))
print("TestSet Score : {:.4f}".format(random_search.score(X_scaled_test, y_test)))
