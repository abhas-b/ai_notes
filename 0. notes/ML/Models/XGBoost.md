#ml [[ML Models]] 

* Builds the trees sequentially
* XGBoost builds the model through the following steps:

1. **Initialize the model**: Start with an initial model F0F_0F0​, typically a constant value like the mean of the target values.
2. **Compute residuals**: Calculate the residuals (errors) of the current model for each sample.
3. **Fit a new tree**: Fit a new decision tree to the residuals.
4. **Update model**: Add the new tree's prediction to the current model using the gradient of the loss function.
5. **Repeat**: Repeat the process iteratively for several boosting rounds until convergence.


Hyperparameters:

- `max_depth`: Controls the depth of trees. Larger values lead to more complex models.
- `learning_rate (eta)`: Controls how much each tree contributes. Smaller values lead to better generalization but require more boosting rounds.
- `n_estimators`: The number of trees to build.
- `subsample`: Fraction of samples used for training each tree, helping with overfitting.
- `colsample_bytree`: Fraction of features used for each tree.

```
import xgboost as xgb

# Convert the data into DMatrix format, which is used by XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set up parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 3,                # Number of classes (3 for the Iris dataset)
    'max_depth': 3,                # Maximum depth of the trees
    'eta': 0.1,                    # Learning rate
    'subsample': 0.8,              # Fraction of samples used for training
    'colsample_bytree': 0.8        # Fraction of features used for training
}

# Train the model with 50 boosting rounds (trees)
num_round = 50
bst = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
preds = bst.predict(dtest)

# Calculate accuracy
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy * 100:.2f}%")
```