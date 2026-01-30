#ml [[Recommender Systems]]

* Instance based learning
* We look at what similar users liked. We find the stuff that particular user liked and what similar users liked for historical data. Then for prediction we find what similar users liked and use it for prediction.


1. User based nearest neighbor
	1. KNN can be used to decide nearest neighbor


Finding KNN using [[numpy]] [[Common Tasks using Python]]

```
dist_sq = np.sum((X[:, np.newaxis,:] - X[np.newaxis, :, :])**2, axis=-1)
nearest = np.argpartition(dist_sq, 2, axis=1)
```

We can also use Pearson's coefficient to find user based neighbors.

$$sim(u,v) = Pearson(u,v)$$

Prediction:
$$p(u,i) = \bar r_u + \frac{\sum sim(u,v) . (r_{v,i} - \bar r_v)}{\sum |sim(u,v)|}$$
Here u is the target user and V denotes k nearest neighbors of u.
We pivot the rating by u user towards his average ratings.

* The issue here is of scalability. When there are many users this is not doable in real time.


2. Item based nearest neighbor
	1. We can find pair wise item similarity and use it for recommendation. If a user likes item a and item b is highly similar to item a then we can recommend item b to this user as well.
	2. This would end up being somewhat similar to [[Content based recommendation systems]]


![[Pasted image 20250129183721.png]]


i and j are items. u is the user. The summation is over all users.


![[Pasted image 20250129183743.png]]
J is the set of similar items.

We basically find the k most similar items. Out of those we do the rating prediction.

