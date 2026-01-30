#ml [[Preprocessing]] 
1. One Hot Encoding
2. Label Encoding

### One Hot Encoding for Independent Variable [[sklearn]] [[ColumnTransformer]]

* If a variable has N unique values, then your model should have at most N-1 dummy variables. The last variable is essentially derivable from the remaining N-1 variables and adding it could lead to multi-collinearity.

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```



### Label Encoding for dependent variable [[sklearn]]

```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
```

