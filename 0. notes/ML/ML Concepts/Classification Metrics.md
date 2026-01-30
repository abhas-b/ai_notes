#ml [[Loss Functions and Metrics]] 

1. Accuracy

$$Accuracy = \frac{\#\ correct\ predictions}{total\ sample\ size}$$
```
from sklearn.model_selection import cross_val_score
cross_val_score(estimator, X, y, cv=3, scoring='accuracy')
```

2. Confusion Matrix

One approach is to make predictions on the test set and build confusion matrix from them.
But since we dont want to touch test set before the launch, we generate cross val predictions and evaluate them.

```
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(estimator, X_train, y_train, cv=5)

sns.heatmap(confusion_matrix(y_train, y_train_pred))
```

TP ------------ FP
FN ------------ TN

3. Precision, Recall, F1-score

Precision: accuracy of positive predictions.

$$Precision = \frac{TP}{TP + FP}$$

Recall:

Also called True Positive Rate (TPR)

Also called Sensitivity

$$Recall = \frac{TP}{TP + FN}$$

Precision:
Accuracy of positive predictions -- 40 out of 100 positive predictions were correct

Recall: 
How many of actual positives were caught by the model.

F1 score: harmonic mean of precision and recall


4. Precision Recall Trade-Off and plotting:

* There are two ways models assign classes to predictions (thereby determining precision/recall):
	* Threshold (for models such as SGD) : decision_function()
	* Class probabilities : predict_proba()

**Increasing threshold ----- lower recall, higher precision**

*sgd.decision_function*

# How to decide threshold for Precision - Recall

## Generate prediction scores on training set using cross val

```
from sklearn.model_selection import cross_val_predict
train_pred_scores = cross_val_predict(clf, X, y, cv=5, method='decision_function')
```

## Draw precision-recall curve

```
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_train, train_pred_scores)

plt.plot(thresholds, precision[:-1])
plt.plot(thresholds, recall[:-1])
```

With the precision-recall curve, we can select a desired level of precision/recall and get corresponding threshold.

We can also use:

`plt.plot(recall, precision)`


5. ROC Curve

Used with binary classifiers.

Plot between FPR and TPR

TPR: True Positive Rate: also called recall

FPR: False Positive Rate

TNR: True Negative Rate : also called specificity

$$FPR = \frac{FP}{FP + TN}$$

$$FPR = 1 - TNR$$

ROC: curve between sensitivity vs 1-specificity


# Plotting ROC

```
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, train_pred_scores)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
```

## Getting AUC:

```
from sklearn.metrics import roc_auc_score
```

Random classifier has AUC = 0.5

Perfect classifier has AUC = 1

# When to use Precision-Recall vs ROC:

* When positive class is rare.



# For RandomForest:

```
y_pred_scores = cross_val_predict(rf_clf, X, y, cv=5, method='predict_proba')
y_pred_scores = y_pred_scores[:, 1] #positive class's probabilities
```

Now we can plot as usual.
