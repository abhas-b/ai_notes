#ml [[ML Concepts]] [[Linear Regression]]

5 methods of building models:
1. All-in
	1. Use all available features in the model.
	2. Useful only if you know that the available set of predictors is exhaustive.
2. Backward Elimination
	1. Select a significance level to stay in the model (SL).
	2. Fit the full model with all the predictors.
	3. Consider the predictor with the highest p-value. If P > SL then remove the predictor. Fit the model without this predictor.
3. Forward Selection
	1. Select a significance level  to enter the model (SL).
	2. Fit all simple regression models (with all features one by one). Select the feature with the lowest P-Value.
	3. Keep this variable and fit all possible models with one extra predictor.
	4. Out of all the possible two feature models, select the predictor with lowest p-value. If P < SL then 
4. Bidirectional Elimination
	1. Select a significance level to enter and to stay in the model (SL Enter, SL Stay)
	2. Perform forward selection (new variables must have p-value < SL Enter to enter the model).
	3. Perform backward elimination for all features added so far.
	4. Repeat 2 and 3.
	5. Once no more churn on features happen then your model is ready.
5. Score Comparison
	1. Build all possible models and evaluate basis a goodness of fit criterion such as AIC, BIC.
	2. This builds 2^ N - 1 models.
	3. 

There's another concept of stepwise regression: this refers to Steps 2,3,4. The default meaning for such cases is bidirectional elimination (unless specified otherwise).


