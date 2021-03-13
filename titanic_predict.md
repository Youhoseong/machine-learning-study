```python
# author: hoseong you 
# titanic predict jupyter notebook code
# CUAI 9기 assignment 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

titanic_df =pd.read_csv('./titanic_train.csv')
titanic_df.head(3)

## 결손 데이터 대체
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True) # NaN을 Age Col의 avg로 대체
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1] # 중요정보만 추출
```


```python
sns.barplot(x='Sex', y='Survived', data=titanic_df) # 성별과 생존의 상관관계 그래프
```




    <AxesSubplot:xlabel='Sex', ylabel='Survived'>




    
![png](output_1_1.png)
    



```python
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df) # 재력, 생존, 성별과의 상관관계
```




    <AxesSubplot:xlabel='Pclass', ylabel='Survived'>




    
![png](output_2_1.png)
    



```python
def get_category(age): # 성별에 따른 카테고리 분류 메소드
    cat = ''
    if age <= -1: 
        cat = 'Unknown'
    elif age <= 5:
        cat= 'Baby'
    elif age <= 12:
        cat= 'Child'
    elif age <= 18:
        cat= 'Teenager'
    elif age <= 25:
        cat= 'Student'
    elif age <= 35:
        cat= 'Young Adult'
    elif age <= 60:
        cat= 'Adult'
    else:
        cat= 'Elderly'
    
    return cat

plt.figure(figsize=(10, 6))
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True) # 그래프 그린 후 dataframe에서 Age_cat col drop = > 그래프만 그리는 용도의 col
        
```


    
![png](output_3_0.png)
    



```python
from sklearn import preprocessing

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature]) # 테스트 데이터에서는 fit 사용하면 안됨
        dataDF[feature] = le.transform(dataDF[feature])
        
    return dataDF

titanic_df = encode_features(titanic_df)
#titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1 , inplace=True) # 한번만 가능한 코드..
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis=1) # 피처, 결정값 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(solver='liblinear')

dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))

```

    DecisionTreeClassifier 정확도: 0.7877
    RandomForestClassifier 정확도: 0.8547
    LogisticRegression 정확도: 0.8659
    


```python
from sklearn.model_selection import KFold

# kfold 교차검증
def exec_kfold(clf, folds=5):
    kfold = KFold(n_splits = folds)
    scores = []
    
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print("교차검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))
    
    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score))



print("# kfold 교차검증")
exec_kfold(dt_clf, folds=5)


#cross_val_score 교차검증

from sklearn.model_selection import cross_val_score
print('\n\n# cross_val_score 교차검증 #')
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))
    
print("평균 정확도: {0:.4f}".format(np.mean(scores)))
```

    # kfold 교차검증
    교차검증 0 정확도: 0.7542
    교차검증 1 정확도: 0.7809
    교차검증 2 정확도: 0.7865
    교차검증 3 정확도: 0.7697
    교차검증 4 정확도: 0.8202
    평균 정확도: 0.7823
    
    
    # cross_val_score 교차검증 #
    교차 검증 0 정확도: 0.7430
    교차 검증 1 정확도: 0.7753
    교차 검증 2 정확도: 0.7921
    교차 검증 3 정확도: 0.7865
    교차 검증 4 정확도: 0.8427
    평균 정확도: 0.7879
    


```python
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth': [2, 3, 5, 10], 'min_samples_split': [2,3,5], 'min_samples_leaf': [1, 5, 8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터:', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))

best_dclf = grid_dclf.best_estimator_

dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 {0:.4f}'.format(accuracy))
```

    GridSearchCV 최적 하이퍼 파라미터: {'max_depth': 3, 'min_samples_leaf': 5, 'min_samples_split': 2}
    GridSearchCV 최고 정확도: 0.7992
    테스트 세트에서의 DecisionTreeClassifier 정확도 0.8715
    
