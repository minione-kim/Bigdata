# 원핫인코딩(one-hot-encoding)
import pandas as pd

data = pd.read_csv("file/vote.csv", encoding='utf-8')

X1 = data[['gender','region']]
XY = data[['edu','income','age','score_gov','score_progress','score_intention','vote','parties']]

X1['gender'] = X1['gender'].replace([1,2], ['male','female'])
X1['region'] = X1['region'].replace([1,2,3,4,5], ['Sudo','Chungcheung','Honam','Youngnam','Others'])

X1_dum = pd.get_dummies(X1)

Fvote = pd.concat([X1_dum,XY], axis=1)

Fvote.to_csv('file/Fvote.csv', index=False, sep=',', encoding='utf-8')

# 2. 데이터셋 분할과 모델검증

# 특성치(x), 레이블(y) 나누기
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
data1 = pd.read_csv('file/Fvote.csv', encoding='utf-8')

# 특성변수(X) 데이터셋을 따로 나눈다
# 방법1 : 특성이름으로 데이터셋 나누기
X = data1[['gender_female', 'gender_male', 'region_Chungcheung', 'region_Honam', 'region_Others', 'region_Sudo', 'region_Youngnam', 'edu', 'income', 'age', 'score_gov', 'score_progress', 'score_intention']]

# 방법2 : 특성 위치값으로 데이터셋 나누기
X2 = data1[data1.columns[0:13]]

# 방법3 : loc 함수로 데이터셋 나누기 (단, 불러올 특성이 연달아 있어야 함)
X3 = data1.loc[:, 'gender_female':'score_intention']

y = data1[['vote']]

# train-test 데이터셋 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 모델 적용
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# 랜덤 없는 교차검증 : cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5)
print("5개 테스트 셋 정확도 : ", scores)
print("정확도 평균 : ", scores.mean())

# 랜덤 있는 교차검증 : K-Fold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(model, X_train, y_train, cv=kfold)
print("5개 폴드의 정확도 : ", score)

# 임의분할 교차검증
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, random_state=42)
score2 = cross_val_score(model, X_train, y_train, cv=shuffle_split)
print("교차검증 정확도 : ", score2)

# train-validity-test 분할과 교차검증
from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, random_state=2)

model.fit(X_train, y_train)
scores2 = cross_val_score(model, X_train, y_train, cv=5)
print("교차검증 정확도 : ", scores2)
print("정확도 평균 : ", scores.mean())

print(model.score(X_valid, y_valid))

print(model.score(X_test, y_test))