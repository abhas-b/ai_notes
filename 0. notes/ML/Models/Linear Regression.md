#ml [[Regression]] 

This model calculates the effect on the dependent variable impact of a unit change in independent variable when the relationship between the two is linear.

Despite being a simple model Linear Regression still allows us to identify the features that have a statistically significant impact on the model.

Simple Linear Regression:
$$Y_i = \beta_0 + \beta_1X_i + \epsilon_i$$
Beta_0 here is called intercept.
Beta_1 is called slope.

Multiple Linear Regression:

$$Y_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{j,i} + \epsilon_i$$

## OLS (Ordinary Least Squares)

Used for parameter estimation.

$$\hat Y_i = \hat \beta_o + \hat \beta_1 X_i$$
$$\text (Residual) \hat u_i = Y_i - \hat Y_i$$



### Assumptions of Linear Regression
1. Linearity: the relationship is linear
	1. Check this by plotting the residuals. This plot should also be linear.
2. No Autocorrelation: Random sample: all observations in the sample are randomly selected. This implies that the residuals are not correlated.
	1. Check: The mean of the residuals should be around 0.
3. Exogeneity: each independent variable is uncorrelated with the error terms.
	1. If this is not true then the estimated coefficients may be biased. We call this problem as endogeneity. 
	2. Endogeneity can arise due to omitted variables (a key predictor is not present in the model), measurement errors, or simultaneous causality.
	3. In a well-specified model with exogenous regressors, it is possible to interpret the coefficients as causal relationships, because the explanatory variables are assumed to be determined outside the model and are not influenced by the error term.
	4. Strict Exogeneity: the assumption holds for all observations.
	5. Weak exogeneity: the assumption holds for the specific observations used to estimate the model and not necessarily over all periods of time or observations.
	6. Check: Hausman test can be used to assess exogeneity. 
	7. Multi-variate normality
		1. Normality of error distribution
4. Homoscedasticity: variance of all error terms is constant across all predictions and not dependent on independent variable.
	1. Tests for the homogeneity of the variance.
	2. Violation can lead to [[Type I errors]].
	3. Check: plot the residuals and check for a funnel like graph
5. No Perfect Multi-collinearity: there are no exact linear relationships between the independent variables.


#### Statistical evaluation: 

 
y = b0 + b1x

line of best fit is obtained through minimizing MSE.

1. **sklearn.linear_model.LinearRegression** [[sklearn]]
	1. Based on least squares from scipy.

```
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X, y, rcond=1e-6)
```

* This is based on pseudoinverse instead of inverse for normal equation.
	* Regular inverse is not defined when no. of features > no. of data points.
	* Pseudoinverse is always defined.

2. **statsmodels.regression.linear_model.OLS** [[statsmodels]]
```
statsmodels.regression.linear_model.OLS(y, X (without intercept), _missing='none'_, _hasconst=None_, _**kwargs_)

results = model.fit()
results.params
results.tvalues

results.t_test([1, 0])

results.f_test(np.identity(2))
```

Add constant thru:

`statsmodels.tools.add_constant`
