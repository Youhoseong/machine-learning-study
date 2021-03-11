```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

iris_data = iris.data

iris_label = iris.target

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df

X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

df_clf = DecisionTreeClassifier(random_state=11)
df_clf.fit(X_train, Y_train)

pred = df_clf.predict(X_test)
print(pred)

print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_test, pred)))
```

    [2 2 1 1 2 0 1 0 0 1 1 1 1 2 2 0 2 1 2 2 1 0 0 1 0 0 2 1 0 1]
    예측 정확도: 0.9333
    
