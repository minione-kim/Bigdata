import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('file/breast-cancer-wisconsin.csv', encoding='utf-8')

# 레이블 변수(유방암) 비율 확인하기
data['Class'].value_counts(sort=False)

# 행(케이스수)과 열(컬럼수) 구조 확인
print(data.shape)

# 방법1 : 특성이름으로 특성 데이터셋(X) 나누기
X1 = data[['Clump_Thickness', 'Cell_Size', 'Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]

# 방법2 : 특성 위치값으로 특성 데이터셋(X) 나누기
X2 = data[data.columns[1:10]]

# 방법3 : loc 함수로 특성 데이터셋(X) 나누기 (단, 불러올 특성이 연달아 있어야 함)
X3 = data.loc[:, 'Clump_Thickness' : 'Mitoses']


# 레이블 데이터셋
y = data[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X1, y, stratify=y, random_state=42)

scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

scaler_standard.fit(X_train)
X_scaled_standard_train = scaler_standard.transform(X_train)

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)

X_scaled_standard_test = scaler_standard.transform(X_test)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

# 모델 - 훈련
model = LogisticRegression()
model.fit(X_scaled_minmax_train, y_train)

pred_train = model.predict(X_scaled_minmax_train)
print("정확도 : ",model.score(X_scaled_minmax_train, y_train))


# 모델 - 테스트
pred_test = model.predict(X_scaled_minmax_test)
# print("정확도 : ", model.score(X_scaled_minmax_test, y_test))

# 오차행렬
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train, pred_train)
# print("훈련데이터 오차행렬:\n", confusion_train)

confusion_test = confusion_matrix(y_test, pred_test)
# print("테스트데이터 오차행렬:\n", confusion_test)

# 레포트
from sklearn.metrics import classification_report
cfreport_train = classification_report(y_train, pred_train)
# print("분류예측 레포트:\n", cfreport_train)

cfreport_test = classification_report(y_test, pred_test)
# print("분류예측 레포트:\n", cfreport_test)

# ROC 지표
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.decision_function(X_scaled_minmax_test))
roc_auc = metrics.roc_auc_score(y_test, model.decision_function(X_scaled_minmax_test))
print(roc_auc)

# ROC Curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1-Specificity')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(false_positive_rate, true_positive_rate, 'b', label = 'Mode (AUC = %0.2f)' %roc_auc)
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')

plt.show()

# 예측값 병합 및 저장
prob_train = model.predict_proba(X_scaled_minmax_train)
y_train['y_pred'] = pred_train
y_train[['y_prob0', 'y_prob1']] = prob_train

prob_test = model.predict_proba(X_scaled_minmax_test)
y_test['y_pred'] = pred_test
y_test[['y_prob0', 'y_prob1']] = prob_test

Total_test = pd.concat([X_test, y_test], axis=1)
print(Total_test)

# 파일 내보내기
Total_test.to_csv("file/classification_test.csv")