#ml [[ML Models]]


* Uses sigmoid function [[Sigmoid]]
* Models the probability of a sample being one of the output classes.

$$\frac{p}{1-p} = b_0 + b_1X_1$$
Loss function:
$$L = y*log(1-p) + (1-y)*log(p)$$

LogisticRegression [[sklearn]]
```
from sklearn.linear_model import LogisticRegression
```