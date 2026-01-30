#ml [[Model Evaluation]]

```
from scipy import stats

confidence_type = 0.95
predictions = model.predict(X_test)

squared_errors = (predictions - X_test)**2
np.sqrt(scipy.t.interval(confidence,
						len(squared_errors) - 1,
						loc=squared_errors.mean(),
						scale = stats.sem(squared_errors)
						))
```


