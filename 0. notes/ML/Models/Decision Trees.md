#ml [[ML Models]]

* Splitting criteria:
	* [[Gini Impurity]]
		* To predict the likelihood that a randomly selected sample would be incorrectly classified.
		* Impurity metric
		* Range: [0, 1]
			* 0 : pure node
			* 0.5: uniform distribution of all classes
	* [[Information Gain]]
		* The feature selected provides most information about a class.
		* It utilizes Entropy
	* [[Entropy]]:
		* Measure of randomness / uncertainty in the data.



* For numeric columns the algo works very slow, since it needs to evaluate each distinct value for node splitting, at each node. Once effective way to speed this up is binning.

* Decision Tree is not suitable if
	* Number of samples is huge
	* Number of features is huge
* Choose [[Random Forest (Bagging)|Random Forest (Bagging)]] in such cases.
* Outliers:
	* Deep trees are more severely impacted by outliers since leaf node would have higher node probability of outlier presence.
* Space complexity: 
$$O(\text{number of nodes})$$
* Run complexity:
$$O(depth)$$

* Training complexity
$$O(nlogn*d)$$


* Can be used with Multi Class Classification
* If you repeat the same categorical feature at different nodes for splitting, you will not get any Information Gain on child nodes. Hence no point reusing categorical features. Numerical feature can be repeated but with different thresholds/cutoffs.
* Feature importance essentially represents Information Gain weighted by % samples impacted by a feature across different nodes and branches.
* For Regression instead of information gain, we compute weighted MSE and want to MSE decrease to be the highest.
