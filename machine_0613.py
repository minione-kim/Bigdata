## 모델평가 ###
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
data = pd.read_csv("file/Fvote.csv", encoding='utf-8')

# 데이터 불러오기 및 데이터셋 분할
X = data[data.columns[1:13]]
y = data[['vote']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.linear_model import LogisticRegression
Final_model = LogisticRegression(C=10)
Final_model.fit(X_train, y_train)

pred_train = Final_model.predict(X_train)
Final_model.score(X_train, y_train)

pred_test = Final_model.predict(X_test)
Final_model.score(X_test, y_test)

# 오차행렬
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
confusion_test = confusion_matrix(y_test, pred_test)

# 분륭예측
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
cfreport_test = classification_report(y_test, pred_test)

# ROC
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, Final_model.decision_function(X_test))
roc_auc = metrics.roc_auc_score(y_test, Final_model.decision_function(X_test))

import matplotlib.pyplot as plt
plt.title("Receiver Operating Characteristic")
plt.xlabel("False Positve Rate(1-Specificity)")
plt.ylabel("True Positive Rate(Sensitivity)")

plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'%roc_auc)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.show()


### 다중분류 ###
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data = pd.read_csv("file/Fvote.csv", encoding="utf-8")

X = data[data.columns[1:13]]
y = data[["parties"]]

# 데이터셋 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 모델 설정
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
model.score(X_train, y_train)

pred_test = model.predict(X_test)
model.score(X_test, y_test)

# 혼동행렬
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
confusion_test = confusion_matrix(y_test, pred_test)

# 그리드탐색
from sklearn.model_selection import GridSearchCV
param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best Parameter : {}".format(grid_search.best_params_))
print("Best Cross-validity Score: {:.3f}".format(grid_search.best_score_))

print("Test set Score : {:.3f}".format(grid_search.score(X_test,y_test)))

# 랜덤탐색
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'C' : randint(low=0.001, high=100)}
random_search = RandomizedSearchCV(LogisticRegression(), param_distributions=param_distribs, cv=5, return_train_score=True)
random_search.fit(X_train, y_train)

print("Best Parameter : {}".format(random_search.best_params_))
print("Best Cross-validity Score:{:.3f}".format(random_search.best_score_))

print("Test set Score : {:.3f}".format(random_search.score(X_test, y_test)))


