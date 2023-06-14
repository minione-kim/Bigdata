### 분류(Classification) ###

# 분석데이터 준비
import warnings
warnings.filterwarnings("ignore")

# 분석데이터 준비
import pandas as pd
data = pd.read_csv("file/breast-cancer-wisconsin.csv", encoding='utf-8')

X = data[data.columns[1:10]]
y = data[['Class']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

# 기본모델 적용
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_scaled_train, y_train)

pred_train = model.predict(X_scaled_train)
# print("정확도 : ", model.score(X_scaled_train, y_train))

pred_test = model.predict(X_scaled_test)
# print("테스트데이터 정확도 : ", model.score(X_scaled_test, y_test))

# 오차행렬
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
# print("훈련데이터 오차행렬 : \n", confusion_train)

confusion_test = confusion_matrix(y_test, pred_test)
# print("테스트데이터 오차행렬 : \n", confusion_test)

# 정밀도, 재현율
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
# print("분류예측 레포트 : \n", cfreport_train)

cfreport_test = classification_report(y_test, pred_test)
# print("테스트예측 레포트 : \n", cfreport_test)

# Grid Search
param_grid = {'n_neighbors' : [1,3,5,7,9,11]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(grid_search.best_params_))
print("Best Score : {:.4f}".format(grid_search.best_score_))
print("TestSet Score : {:.4f}".format(grid_search.score(X_scaled_test, y_test)))

# Random Search
from scipy.stats import randint
param_distribs = {'n_neighbors' : randint(low=1, high=20)}
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distribs, n_iter=20, cv=5)
random_search.fit(X_scaled_train, y_train)

print("Best Parameter : {}".format(random_search.best_params_))
print("Best Score : {:.4f}".format(random_search.best_score_))
print("TestSet Score : {:.4f}".format(random_search.score(X_scaled_test, y_test)))


