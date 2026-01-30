#ml [[ML Concepts]]


* Bias
	* Inability of a model to capture the true relationship in the data.

$$Bias = E(\hat f(x_0)) - f(x_0)$$


* Variance is the inconsistency level of model performance when applying to different data sets.
* If the model trained on one data set performs drastically different on another dataset then we say that variance is high.


$$Var = E[(\hat f(x_0) - E(\hat f(x_0)))^2]$$



* Prediction Error Rate

$$ E(y_0 - \hat f(x_0))^2 $$
$$E(Y-Y)^2 = E[f(X) + \epsilon - \hat f(X)$$
$$ E(Y-Y)^2 = [f(X) - \hat f(X)]^2 + Var(\epsilon)$$
$$ Var (\epsilon) \text{is the irreducible component of ther error rate}$$



We can summarize error rate as:
$$E(Y - \hat Y)^2 = Var(\hat f(x_0)) + [Bias(\hat f(x_0))]^2 + Var(\epsilon)$$


# [[Overfitting and Regularization]]

* Overfitting
	* Low train error rate but high test error rate
	* High variance low bias
* Underfitting
	* High train error rate but low test error rate
	* High bias


## Fixing overfitting
* Reduce model complexity
* More data
* Early stopping
* Use resampling techniques of Cross Validation
* Ensemble methods
* Dropout


## Regularization
* Shrinks some of the model coefficients to 0 to penalize unimportant variables.
* Introduces a little bias in the model while reducing variance.

### Ridge Regression (L2 norm)
* It doesn't force the parameters to be exactly 0.
* This also leads to a reduction in interpretability in cases where number of features are fairly large.

$$ Loss function = \sum_{i=1}^{n}(y_i - \beta_0 - \sum_{j=1}^{p}\beta_jx_{ij})^2 + \lambda\sum_{j=1}^{p}\beta_j^2$$

### Lasso Regression (L1 norm)
* Shrinks some parameters to be exactly 0.
* This also leads to feature selection as some features are shrunk to 0.
* 

$$ Loss function = \sum_{i=1}^{n}(y_i - \beta_0 - \sum_{j=1}^{p}\beta_jx_{ij})^2 + \lambda\sum_{j=1}^{p}|\beta_j|$$

### Dropout (Neural Nets)

