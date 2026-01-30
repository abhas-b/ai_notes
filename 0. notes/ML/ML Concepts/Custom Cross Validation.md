#ml [[Cross Validation]]
Steps to implement cross validation:
* split training data into folds
* train fresh model on each fold and evaluate on each test fold (run K-Fold)

```
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(cv=5, random_state=42)
og_model = SGDClassifier()

for train_index, test_index in skfolds.split(X, y):
	model = clone(og_model)
	model.fit(X[train_index], y[train_index])
	
	y_preds = model.predict(X[test_index])
	score = sum(y_preds == y[test_index])
	print(score / len(y_preds))
```