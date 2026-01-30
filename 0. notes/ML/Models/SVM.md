#ml [[Regression Methods]]

* Tries to find the best fit line that has the largest margin. 
* Not only we want to separate the classes well, we also care about the boundary in between the data points and the line that we're drawing. The distance between the boundary and the nearest points from both classes is called margin.
* The points on the margin are called support vectors.
* SVMs are not very robust to outliers.
* SVMs use what is known as kernel trick. They take projections of the data onto a hyperplane where its easier to draw the line. An example could be a squared kernel $$x^2$$
* For SVM we care about the distance of support vectors with the edge of margin.
```
from sklearn.svm import SVC
```
