#ml [[ML Models]]


* De-correlates the individual decision trees.
* At each split a random subset of the data is used to train a tree based on a random subset of features as well.  The sample size of predictors is usually 

$$\sqrt (\text {total number of features}) $$

* Random Forest is a good model to deal with multi-collinearity.

* Makes use of Random Sampling with replacement. Even a single sample can have repeated / duplicate rows due to replacement.
* It does random feature selection as well as random sampling. Combined together, they are called random **subspacing**.


* Feature selection:
	* Classification:
		* No. of features = sqrt(total number of features)
	* Regression:
		* No. of features = total number of features / 3
	


* Trains multiple models on different subsets of data (sampling with replacement) and combines predictions through averaging or voting.
	* Do both row as well as column sampling (sqrt p columns are used).
* This diversity in individual predictors reduces variance and bias especially in unstable models like decision trees.

Random Forest [[sklearn]]:

```
from sklearn.ensemble import RandomForestClassifier
```

## Out of Bag Samples
* Samples for any tree remaining after subspacing are called Out of Bag samples. Also called OOB.
* OOB samples can be used for [[Model Evaluation]]
* The errors on OOB samples is called OOB errors.



Another advantage of Random Forests is that due to sub-sampling it uses lesser data for each tree training. Hence every tree sees lesser data than even a full Decision Tree Model. Hence resource requirement is lesser as well as faster.
Random Forest is also parallelized. Hence overall computation time is lesser than k x number of trees. Hence Random Forest is well suited for Distributed Computing.

### Feature Importance:

* Since Random Forest performs column sampling, each column gets consumed by approximately equal number of trees. Hence we can get Information Gain from each feature from each tree, add it and then average it over total number of features.
* This aggregation gives us information gain for a feature across the forest.
* The features with highest aggregate IG are the most important features.


### Hyperparameter Tuning

* Higher number of trees implies samples becoming smaller. Hence the trees become even more uncorrelated. Hence higher number of trees is desirable.
* Row sampling: typically 20%-40%
* Depth of tree
	* Suggested depth: 60% of number of features


### Regularization
$$min(loss) + \lambda(leaves)$$








