#ml [[Regression Methods]]

$$Y_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{j,i}^j + \epsilon_i$$

* Polynomial Regression allows us to fit a linear model on non-linear data.

Polynomial Features [[sklearn]]
```
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
```

