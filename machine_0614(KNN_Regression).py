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
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_scaled_train, y_train)
pred_train = model.predict(X_scaled_train)
print("정확도 : ", model.score(X_scaled_train, y_train))

pred_test = model.predict(X_scaled_test)
print("테스트 정확도 : ", model.score(X_scaled_test, y_test))

# RMSE (Root Mean Squared Error)
import numpy as np
from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, pred_train)
MSE_test = mean_squared_error(y_test, pred_test)
print("훈련데이터 RMSE : ", np.sqrt(MSE_train))
print("테스트데이터 RMSE : ", np.sqrt(MSE_test))

# GridSearch
param_grid = {'n_neighbors' : [1,3,5,7,9,11]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(grid_search.best_params_))
print("Best Score : {:.4f}".format(grid_search.best_score_))
print("TestSet Score : {:.4f}".format(grid_search.score(X_scaled_test, y_test)))

# RandomSearch
from scipy.stats import randint
param_distribs = {'n_neighbors' : randint(low=1, high=20)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(KNeighborsRegressor(), param_distribs, n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(random_search.best_params_))
print("Best Score : {:.4f}".format(random_search.best_score_))
print("TestSet Score : {:.4f}".format(random_search.score(X_scaled_test, y_test)))
