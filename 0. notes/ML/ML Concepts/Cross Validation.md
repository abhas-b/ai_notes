#ml [[ML Concepts]]



* Regression scoring using MSE: 'neg_mean_squared_error'

## Cross Val [[sklearn]]

```
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, scoring='neg_mean_squared_error', cv=10)
scores *= -1

score_rmse_scores = np.sqrt(scores)
overall_rmse = np.mean(score_rmse_scores)
std_rmse = np.std(score_rmse_scores)
```


This gives us a score for each fold. We can check these out for distribution properties such as mean/std.

* We can use these results to compare with training set results to check for overfitting.
* During tuning, we can evaluate these results for each set of hyperparameters to find out optimum parameters.


## K Fold [[sklearn]]
```
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

n_folds = 5

param_grid = {'max_depth':range(1,40),
			  'min_samples_leaf':range(50,150,10)
			  }

tree = GridSearchCV(classifier, param_grid=param_grid, cv=n_folds, scoring="accuracy")
tree.fit(X,y)

scores = tree.cv_results_
tree.best_score_
tree.best_estimator_
```

```
param_grid = [
				{'param_1': [val1, val2, ....]},
				{'param_1': [val1, val2, ....], 
					'param_2': [val1, val2, ...]}
			]
grid_search = GridSearchCV(estimator, 
							param_grid, 
							cv=5, 
							scoring="neg_mean_squared_error", 
							return_train_score=True)
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_
```

* After finding the best estimator, grid search retrains the estimator. Hence the final grid_search object is the trained model with the best parameters.

```
evaluation_scores = grid_search.cv_results_
for mean_score, params in zip(evaluation_scores['mean_test_score'], evaluation_score['params']):
	print(np.sqrt(-mean_score), params)
```

Hyperparameters in custom transformers can also be passed to grid search for optimization.

