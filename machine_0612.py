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
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True)
grid_search.fit(X_train, y_train)

print("Best Parameter:{}".format(grid_search.best_params_))
print("Best Cross-validity Score:{:.3f}".format(grid_search.best_score_))
print("Test set Score:{:.3f}".format(grid_search.score(X_test, y_test)))

result_grid = pd.DataFrame(grid_search.cv_results_)

import matplotlib.pyplot as plt
plt.plot(result_grid['param_C'], result_grid['mean_train_score'], label='Train')
plt.plot(result_grid['param_C'], result_grid['mean_test_score'], label="Test")
plt.legend()
# plt.show()

# Random Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribs = {'C':randint(low=0.001, high=100)}

from sklearn.linear_model import LogisticRegression
random_search = RandomizedSearchCV(LogisticRegression(),
                                   param_distributions=param_distribs, cv=5,
                                   # n_iter=100, 랜덤횟수 디폴트=10
                                    return_train_score=True)
random_search.fit(X_train, y_train)

print("Best Parameter:{}".format(random_search.best_params_))
print("Best Cross-validity Score:{:.3f}".format(random_search.best_score_))
print("Test set Score:{:.3f}".format(random_search.score(X_test, y_test)))

result_random = pd.DataFrame(random_search.cv_results_)

import matplotlib.pyplot as plt
plt.plot(result_random['param_C'], result_random['mean_train_score'], label="Train")
plt.plot(result_random['param_C'], result_random["mean_test_score"], label="Test")
plt.legend()
# plt.show()