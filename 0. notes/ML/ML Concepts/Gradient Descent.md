#ml [[ML Concepts]] 

* General form of linear regression:

$$\hat{y} = h_{\theta}(x) = \theta^Tx$$
such that x0 = 1

h(x) is called the hypothesis function.

$$MSE(X, h_{\theta}) = \frac{\sum_{i=1}^{m} (h(x)^i - y^i)^2}{m}$$
Works in an iterative manner to reduce the cost by tweaking the parameters.

Challenges with gradient descent:
* Initialization: 
	* it can lead convergence to a local minimum.
	* it can make the convergence very slow if learning rate is too small
* Learning rate selection
	* Decides whether or not convergence occurs and whether global minimum is found.


* The cost function for Linear Regression is convex : there are no local minimum.
	* Its slope never changes abruptly.

Gradient Descent works best when features are scaled. It ensures faster convergence.


## Batch Gradient Descent

Partial derivative of cost function:

$$\frac{\partial}{\partial \theta_j} MSE = \frac{2}{m} \sum_{i=1}^{m} (\theta^Tx^{(i)} - y^{(i)})x_j^{(i)}$$

Vectorized version of this equation:

$$\Delta MSE(\theta) = \frac{2}{m} X^T(X\theta - y)$$

Since this version uses entire dataset to evaluate MSE, it is called Batch Gradient Descent.

$$\theta^{\text{next step}} = \theta - \eta \Delta MSE(\theta)$$

### Finding Learning rate:
* Use [[Grid Search]]

### Finding number of iterations:
* Use [[Early Stopping]]
	* When change in cost becomes less than a threshold / tolerance, we can stop.



## Stochastic Gradient Descent

Picks a random instance from the dataset and computes cost on that to estimate the next step.

The stochastic nature of the algo allows it to (sometimes) escape the local minima.

* Learning rate scheduling: reducing the learning rate gradually
	* It helps the algo avoid local minima initially due to large lr, while lets it settle close to global minima due to small lr.

One trick for SGD is to shuffle dataset first and then apply SGD. This needs to be at the start of each epoch. This ensures that data points are not evaluated multiple times.

```
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter=1000, penalty=None, tol=1e-6, eta=0.1)
```






