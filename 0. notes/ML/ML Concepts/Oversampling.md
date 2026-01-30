#ml [[Preprocessing]]

```
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X,y = ros.fit_resample(X,y)
```
