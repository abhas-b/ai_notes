
#ml [[ML Concepts]] [[Model Evaluation]]

Types of loss functions:

1. L1 loss function
	1. It is robust to outliers because it doesn't square the errors, unlike L2 loss. 
	2. The gradient of L1 loss is constant for all non-zero differences, making it less sensitive to large errors.

$$\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
2. L2 loss function
	1. The L2 loss function is **differentiable** and **convex**, making it suitable for optimization with gradient-based methods like gradient descent.
	2. It is sensitive to **outliers**, as the square of large errors increases disproportionately.
	3. L2 loss is widely used in: 
		- **Linear regression**: where the goal is to minimize the error between predicted and actual values. 
		- **Neural networks**: in regression tasks or in the output layer of networks used for regression.


 $$ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3. Binary cross entropy loss (also called log loss)
	1. Used for binary classification problems.
	2. Uses:
		-  **Sigmoid Activation**: The Binary Cross Entropy loss is typically used in conjunction with a **sigmoid activation** function in the output layer, which squashes the output to the range \( [0, 1] \), representing probabilities.
		- The BCE loss function is **differentiable** and can be optimized efficiently using gradient descent. 
		- It is sensitive to **misclassified points** with high confidence, i.e., if the model is very confident but wrong, the penalty is very large.

$$\text{BCE}_{\text{avg}} = - \frac{1}{n} \sum_{i=1}^{n} \left( y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right)$$
4. Accuracy

$$ \frac{\text {Correct Predictions}}{\text {All Predictions}}$$
5. Precision
	1. Focusses on the quality of the positive predictions made by the model. It is a measure of how many of the items that the model classified as positive are actually positive.
	2. **Percentage of positive predictions that are actually positive.**
	3. Important in fraud detection, medical diagnoses, spam filtering etc. In such scenarios the datasets are often imbalanced with higher number of negative samples.
	4. Useful when we want to minimize false positives.


$$ \frac{TP}{TP + FP} $$


6. Recall
	1. Also called sensitivity or True Positive Rate (TPR)
	2. Focuses on the ability of the model to correctly identify positive instances. It measures how well the model captures all the actual positive cases.
	3. **Percentage of positives actually captured.**
	4. Useful when we want to minimize false negatives.


$$\frac{TP}{TP + FN}$$

7. F1 Score
$$ \frac{2 * Recall * Precision}{Recall + Precision}$$
8. RSS (Residual Sum of Squares)

$$\sum_{i=1}^{N}(y_i - \hat y)^2$$
9. MSE
	* Sensitive to outliers

$$\frac{\sum_{i=1}^{N}(y_i - \hat y)^2}{N}$$
10. RMSE
	* More interpretable since its in the same units as your data.
	* RMSE places higher emphasis on samples with larger errors.
	* Performs rather poorly when there are many outliers in the data.
	* In such a case, MAE should be preferred.
	* sklearn.metrics.mean_squared_error [[sklearn]]
	* sklearn.metrics.mean_absolute_error [[sklearn]]


$$\sqrt(MSE)$$

11. MAE
	* Less sensitive to outliers
$$\frac{\sum_{i=1}^{N}|y_i - \hat y|}{N}$$



12. R-Squared

$$R^2 = 1 - \frac{SSR}{SST}$$
$$R^2 = 1- \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y_i)^2}$$

* SSR: sum of squared regression
* SST: total sum of squares

Ideal R Squared: 1

13. Adjusted R-Squared
* Issue with R squared: simply increasing number of features (even if irrelevant features) will increase R squared.
$$Adj R^2 = 1 - \frac{(1 - R^2)(N-1)}{N - p - 1}$$
R^2: R Squared
N : Sample size
p: number of features

* It will be generally lower than R squared.
## Clustering Performance Metrics

* Homogeneity

$$h = 1 - \frac{\text {Conditional entropy given cluster assignments}}{\text {Entropy of predicted class}}$$

* Silhouette Score
	* Higher score indicates that the data points is well matched with its cluster.
	* Usually used with DBSCAN or KMeans.

$$s(o) = \frac{b(o) - a(o)}{max \{a(o) - b(o)\}}$$


* Completeness
$$c = 1 - \frac{\text {Conditional entropy given cluster assignments}}{\text {Entropy of actual class}}$$






