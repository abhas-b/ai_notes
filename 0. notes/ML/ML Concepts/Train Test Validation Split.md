#ml [[ML Concepts]] [[Common Tasks using Python]]

```
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
```

Apply [[Feature Scaling]] after splitting the dataset.

Train Test Split [[sklearn]]

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
```