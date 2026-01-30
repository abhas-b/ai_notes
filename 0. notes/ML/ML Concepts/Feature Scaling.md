#ml [[Preprocessing]]


* Not to be applied to the dummy variables obtained from [[Encoding Categorical Data]]
## Standardization

* Works well in general.

$$X = \frac{X - \mu}{\sigma}$$

```
from sklearn.preprocessing import StandardScaler

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values

scaler = StandardScaler()
X = scaler.fit_transform()

data = np.hstack([X,np.reshape(y, (-1,1))])
```
* Generally the range would be [-3,3]. Anything outside can be generally considered as outlier.

## Normalization
* Recommended when you have a normal distribution in most of your features.

$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

* Range: [0,1]