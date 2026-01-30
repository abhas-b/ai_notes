#ml [[ML Models]]

* Combine many weak models to create a powerful model.
* The trees are built sequentially, each focusing on correcting the previous model's mistakes.
* Doesn't use random data samples like bagging.
* We use small trees (high bias) (depth: between 1 and 4). 



## AdaBoost [[Adaboost]]

* Start with equal weights for all data points.
* Build a stump choosing the best single feature for splitting.
	* Determined by RSS of Gini index [[Loss Functions and Metrics]]
* Calculate the stump's error, use it to determine its importance (weight).
* Update the data point weights. Harder to classify data points get higher weights.
* Continue building stumps based on updated weights.
* All the stumps are then combined taking into account their importance (weights).
* Can be used to arrive at feature importance.



## GBM - Gradient Boosting Machines

* Combines weak decision trees sequentially, each tree aiming to correct the errors (residuals) of the combined model so far.
* Calculates gradients of the loss function telling the algo how to adjust predictions to minimize error.
* Also has a learning rate to minimize overfitting.
* Also supports early stopping.


Steps:
1. Train the model on existing data to predict the outcome variable.
2. Compute the error rate using the predictions and real values. (Loss function) (this gives pseudo residual)
3. Use the existing features and pseudo residual as the outcome variable to predict the residuals again.
4. Use the predicted residuals to update the predictions from Step 1 while scaling this contribution to the tree with a learning rate.
5. Repeat Steps 1-4 until stopping criteria arrives.


## XGBoost

Step 1: assume a mean model. Compute error with mean as predictions to get error_0 using a shallow Decision Tree.
Step 2: Use x and error_0 as inputs to another Decision Tree and repeat Steps 1 and 2.

* Prone to overfitting if not regularized.

After m iterations. the loss function becomes:
$$F_m(x) = \sum_{i=0}^{m}\alpha_ih_i(x)$$
where alpha represents learning rate.
h_i(x) represents model hypothesis for ith iteration.

