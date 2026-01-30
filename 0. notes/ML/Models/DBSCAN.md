#ml [[Clustering]]

* Density Based Spatial Clustering of Applications with Noise
* If the actual data is nested, which means the data is all over the place in such a way that one cluster surrounds the other one, K-Means will have trouble segregating the two.
* Uses the densities of the points in the clusters. 
* It'll identify a random point and within a user defined parameter for radius, within the circle of this radius all points will be identified within the same cluster. 
* We define something called a core point.
	* This is a data point that is close to atleast X number of data points where X is user defined.
* After identifying all core points, we randomly pick a core point and assign it to a cluster.
* All its close points are also added to the same cluster.
* Keep extending the cluster as long as you keep finding core points. Then add the non core points close to the core points to the cluster. Non core points wont be used to extend further for close points.
* Points which don't get assigned to any cluster are called outliers.
* Struggles with very high dimensional data.

DBSCAN [[sklearn]]
```
model = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5)
model.fit_predict(X)
```
