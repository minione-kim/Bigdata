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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

logit_model = LogisticRegression(random_state=42)
rnf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(random_state=42)

voting_hard = VotingClassifier(
    estimators=[('lr', logit_model), ('rf', rnf_model), ('svc', svm_model)],
    voting = 'hard')
voting_hard.fit(X_scaled_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (logit_model, rnf_model, svm_model, voting_hard):
    clf.fit(X_scaled_train, y_train)
    y_pred = clf.predict(X_scaled_test)
    # print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# 혼동행렬 - 로지스틱
from sklearn.metrics import confusion_matrix
log_pred_train = logit_model.predict(X_scaled_train)
log_confusion_train = confusion_matrix(y_train, log_pred_train)
print("로지스틱 분류기 훈련데이터 오차행렬 : \n", log_confusion_train)

log_pred_test = logit_model.predict(X_scaled_test)
log_confusion_test = confusion_matrix(y_test, log_pred_test)
print("로지스틱 분류기 테스트데이터 오차행렬 : \n", log_confusion_test)

# 혼동행렬 - 서포트벡터머신
svm_pred_train = svm_model.predict(X_scaled_train)
svm_confusion_train = confusion_matrix(y_train, svm_pred_train)
print("SVM 분류기 훈련데이터 오차행렬 : \n", svm_confusion_train)

svm_pred_test = svm_model.predict(X_scaled_test)
svm_confusion_test = confusion_matrix(y_test, svm_pred_test)
print("SVM 분류기 테스트데이터 오차행렬 : \n", svm_confusion_test)

# 혼동행렬 - 랜덤포레스트
rnd_pred_train = rnf_model.predict(X_scaled_train)
rnd_confusion_train = confusion_matrix(y_train, rnd_pred_train)
print("랜덤포레스트 훈련데이터 오차행렬 : \n", rnd_confusion_train)

rnd_pred_test = rnf_model.predict(X_scaled_test)
rnd_confusion_test = confusion_matrix(y_test, rnd_pred_test)
print("랜덤포레스트 테스트데이터 오차행렬 : \n", rnd_confusion_test)

logit_model = LogisticRegression(random_state=42)
rnf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)

voting_soft = VotingClassifier(
    estimators=[('lr',logit_model), ('rf', rnf_model), ('svc',svm_model)],
    voting='soft')
voting_soft.fit(X_scaled_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (logit_model, rnf_model, svm_model, voting_soft) :
    clf.fit(X_scaled_train, y_train)
    y_pred = clf.predict(X_scaled_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
voting_pred_train = voting_soft.predict(X_scaled_train)
voting_confusion_train = confusion_matrix(y_train, voting_pred_train)
print("투표분류기 훈련데이터 오차행렬 : \n", voting_confusion_train)

voting_pred_test = voting_soft.predict(X_scaled_test)
voting_confusion_test = confusion_matrix(y_test, voting_pred_test)
print("투표분류기 테스트데이터 오차행렬 : \n", voting_confusion_test)