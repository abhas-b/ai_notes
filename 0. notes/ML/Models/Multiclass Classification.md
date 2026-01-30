#ml [[ML Models]] 

* Native handling:
	* [[Random Forest (Bagging)]]
	* [[Stochastic Gradient Descent]]
	* [[Naive Bayes]]
* Strictly binary classifiers:
	* [[Logistic Regression]]
	* [[SVM - beta]]


* Multiclass strategies:
	* OvR: One vs Rest
		* Also called One vs All
		* Train K binary classifiers
		* For a data point to predict, make predictions from all K classifiers, and follow the one with the highest score.
	* OvO: One vs One
		* Train a binary classifier for each pair of classes.
		* N(N-1)/2 classifiers would be trained
		* Preferred for algos that scale poorly with large scale data, since each individual classifier would run on 2K/N data points instead of N.
		* For each class, we get one score. Class prediction corresponds to the highest among these.


