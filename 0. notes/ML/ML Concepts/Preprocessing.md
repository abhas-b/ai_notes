#ml [[ML]]


1. [[Missing Data]]
2. [[Encoding Categorical Data]]
3. [[Create a pre-processing template class]]
4. [[Train Test Validation Split]]
5. [[Attribute Combinations]]
6. [[Feature Scaling]]
7. [[Custom Transformers]]
8. [[Pipelining]]


We can analyze correlations between the variables to understand the data better, although this might lead to inaccurate conclusions, since correlation is a metric of linear relationship.

```
plt.plot(df.corr())
```

```
from pandas.plotting import scatter_matrix
scatter_matrix(df)
```





