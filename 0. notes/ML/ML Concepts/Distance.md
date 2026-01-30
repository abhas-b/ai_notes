#ml [[ML Concepts]]

Types of distances:

1. Euclidean Distance / Norm
	1. RMSE is essentially Euclidean distance
	2. Also called l2-norm
	$$L2 - norm = ||x||_2$$
2. Manhattan norm:
	1. Also called L1-norm and MAE
	2. Measures distance between two points in a city if you are allowed to travel only horizontally and vertically (orthogonally).

$$L1 - norm = ||x||_1$$
3. L-k norm:
	1. At k = 0, it represents the number of non-zero elements in the vector (due to limit).
	2. At k = inf, it represents the maximum absolute value in the vector.
	3. For higher k values, L-k norm focusses more and more on higher absolute values. That's why l2 norm is affected more severely by outliers compared to l1 norm.

	$$||v||_k = (|v_0|^k + |v_1|^k + ...+|v_n|^k)^{1/k}$$
4. 	