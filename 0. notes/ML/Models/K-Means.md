#ml 

1. Decide how many clusters you want.
2. For each cluster randomly place a centroid.
3. K Means then assigns points closest to each centroid to its respective cluster. 
4. We calculate the center of mass (or center of gravity) for each of the clusters.  
	1. For each cluster calculate average(x) and average(y)
	2. Move the centroid to the x mean and y mean as calculated above.
5. Repeat 2-4 until we get stable centroids.


### Elbow Method

* Used for determining number of clusters.
* We calculate Within Cluster Sum of Squares [[Model Evaluation]]
$$WCSS = \sum_{P_i \text { in cluster 1}} distance(P_i, C_1)^2 + \sum_{P_i \text { in cluster 2}} distance(P_i, C_2)^2 + ...$$

* We first run K Means, then we calculate WCSS.
* Increasing number of clusters reduces WCSS.
* When number of clusters >= number of points: WCSS = 0 (each data point becomes its own centroid)

### Random Initialization Trap

* Because the centroids are initialized randomly
	* They can give different clusters on each run
	* Points may get assigned to wrong clusters simply due to randomness in initialization.
	* Solution: K-Means ++
		* Choose the first centroid at random among the data points
		* For each of the remaining data points compute the distance D to the nearest out of already selected centroids.
		* Choose next centroid among remaining data points using weighted random selection - weighted by D^2
		* Repeat steps until all centroids have been selected.
	* Proceed with K-Means

K-Means [[sklearn]]
```
kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
```

```
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
