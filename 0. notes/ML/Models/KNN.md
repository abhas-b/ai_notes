#ml  [[ML Models]]

* Basically looks around to find labels for its neighbors and assigns the same label.
* A core component is a distance function
	* Euclidean distance$$ \sum_{i=1}^{n} (x_1 - x_2)^2$$

KNN [[sklearn]]

```
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

```


#### Classification Report [[Model Evaluation]][[sklearn]]

```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

