#ml [[ML Models]]

* Based on [[Bayes' Rule]]


![[Pasted image 20241211200441.png]]


![[Pasted image 20241211200600.png]]


* The Naive part here is that all features are independent.

![[Pasted image 20241211201226.png]]

![[Pasted image 20241211201457.png]]

* This is known as MAP (Maximum A Posteriori)



NaiveBayes [[sklearn]]
```
from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
```

* It can be a high bias model due to its simplicity.
* 