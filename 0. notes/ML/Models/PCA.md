#ml 

Applications:
* Noise reduction
	* Removes correlated features
* Visualization
* Feature Extraction


Projects high dimensional data onto a smaller dimensional subspace.

1. Standardize the data
2. Obtain eigenvalues and eigenvectors from the covariance matrix (perform Singular Value Decomposition)
3. Sort eigenvalues in descending order and choose the k eigenvectors that correspond to k largest eigenvalues (corresponding to desired number of features in smaller subspace)
4. Construct the projection matrix W from the selected k eigenvectors.


* Highly affected by outliers in the data.

