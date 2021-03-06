Highly interpretable, sklearn-compatible classifier and regressor based on simplified decision trees
===============

Implementation of a simple, greedy optimization approach to simplifying decision trees for better interpretability and readability. 

It produces small decision trees, which makes trained classifiers **easily interpretable to human experts**, and is competitive with state of the art classifiers such as random forests or SVMs.

Turns out to frequently outperform [Bayesian Rule Lists](https://github.com/tmadl/sklearn-expertsys) in terms of accuracy and computational complexity, and Logistic Regression in terms of interpretability.
Note that a feature selection method is highly advisable on large datasets, as the runtime directly depends on the number of features. 

Usage
===============

The project requires [scikit-learn](http://scikit-learn.org/stable/install.html).

The included `InterpretableDecisionTreeClassifier` and `InterpretableDecisionTreeRegressor` both work as scikit-learn estimators, with a `model.fit(X,y)` method which takes training data `X` (numpy array or pandas DataFrame) and labels `y`.

The learned rules of a trained model can be displayed simply by casting the object as a string, e.g. `print model`, or by using the `model.tostring(feature_names=['feature1', 'feature2', ], decimals=1)` method and specifying names for the features and, optionally, the rounding precision. 

Example output on `breast cancer` dataset:

```python
# Data from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
def breast_cancer_probability(radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension):
  if perimeter <= 2.5:
    if concavity <= 5.5: return 0.012
    else: return 0.875
  else:
    if area <= 2.5: return 0.217
    else: return 0.917
```

Tree size and complexity can be reduced by two parameters: 
* the classical [`max_depth` parameter](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier), and
* the `acceptable_score_drop` parameter, which specifies the maximum acceptable reduction in classifier performance (higher means more branches can be pruned). By default, the F1-score is used for this purpose. A `scorer` parameter can be passed to the `fit` method if optimization based on other scores is preferred. 

Self-contained usage example:

```python
import numpy as np
from sklearn.datasets.samples_generator import make_moons
from sklearn.model_selection._validation import cross_val_score
from InterpretableDecisionTreeClassifier import *

X, y = make_moons(300, noise=0.4)
print("Decision Tree F1 score:",np.mean(cross_val_score(DecisionTreeClassifier(), X, y, scoring="f1")))
print("Interpretable Decision Tree F1 score:",np.mean(cross_val_score(IDecisionTreeClassifier(), X, y, scoring="f1")))

"""
**Output:**
Decision Tree F1 score: 0.81119342213567125
Interpretable Decision Tree F1 score: 0.8416950113378685
"""
```

![Simplified decision tree on moons dataset](example_dt.png)

Comparison with other sklearn classifiers (can be reproduced with `run_demo_classifier_comparison.py'. Rule List Classifier: see [here](https://github.com/tmadl/sklearn-expertsys))

```python
                       D.Tree3 F1          D.Tree5 F1            Interpr.D.Tree3 F1      Interpr.D.Tree5 F1     RuleListClassifier F1   Random Forest F1      
==========================================================================================================================================================
diabetes_scale        0.814 (SE=0.006)    0.808 (SE=0.007)        0.826 (SE=0.005)       *0.833 (SE=0.005)      0.765 (SE=0.007)        0.793 (SE=0.006)
breast-cancer         0.899 (SE=0.005)    0.912 (SE=0.005)        0.920 (SE=0.004)        0.917 (SE=0.004)      0.938 (SE=0.004)       *0.946 (SE=0.004)
uci-20070111 haberman 0.380 (SE=0.020)    0.305 (SE=0.019)        0.380 (SE=0.020)       *0.404 (SE=0.015)      0.321 (SE=0.019)        0.268 (SE=0.017)
heart                 0.827 (SE=0.005)    0.800 (SE=0.005)        0.824 (SE=0.005)       *0.828 (SE=0.006)      0.792 (SE=0.006)        0.808 (SE=0.008)
liver-disorders       0.684 (SE=0.013)    0.610 (SE=0.017)       *0.702 (SE=0.014)        0.670 (SE=0.016)      0.663 (SE=0.019)        0.635 (SE=0.016)

==== Interpretable DT for dataset `diabetes_scale' (lines of full DT: 24, lines of interpretable DT: 6, simplification factor: 0.25) ====
def probability_of_class_one(ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7):
  if ft1 <= 0.2814: return 0.8062
  else:
    if ft5 <= -0.1073: return 0.6842
    else: return 0.2754

==== end of DT for dataset `diabetes_scale'. F1 score: 0.835061262959 ====

==== Interpretable DT for dataset `breast-cancer' (lines of full DT: 24, lines of interpretable DT: 8, simplification factor: 0.333333333333) ====
def probability_of_class_one(ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9):
  if ft2 <= 2.5:
    if ft6 <= 5.5: return 0.0122
    else: return 0.875
  else:
    if ft3 <= 2.5: return 0.2174
    else: return 0.9174

==== end of DT for dataset `breast-cancer'. F1 score: 0.936605316973 ====

WARNING: No target found. Taking last column of data matrix as target
==== Interpretable DT for dataset `uci-20070111 haberman' (lines of full DT: 21, lines of interpretable DT: 10, simplification factor: 0.47619047619) ====
def probability_of_class_one(ft0, ft1, ft2):
  if ft2 <= 4.5:
    if ft0 <= 77.5: return 0.1754
    else: return 1.0
  else:
    if ft0 <= 42.5:
      if ft2 <= 20.5: return 0.0833
      else: return 0.6667
    else: return 0.5902

==== end of DT for dataset `uci-20070111 haberman'. F1 score: 0.544217687075 ====

==== Interpretable DT for dataset `heart' (lines of full DT: 24, lines of interpretable DT: 12, simplification factor: 0.5) ====
def probability_of_class_one(ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12):
  if ft12 <= 4.5:
    if ft2 <= 3.5: return 0.901
    else:
      if ft11 <= 0.5: return 0.8065
      else: return 0.15
  else:
    if ft11 <= 0.5:
      if ft8 <= 0.5: return 0.6897
      else: return 0.2083
    else: return 0.0923

==== end of DT for dataset `heart'. F1 score: 0.87459807074 ====

==== Interpretable DT for dataset `liver-disorders' (lines of full DT: 24, lines of interpretable DT: 6, simplification factor: 0.25) ====
def probability_of_class_one(ft0, ft1, ft2, ft3, ft4, ft5):
  if ft4 <= 20.5:
    if ft2 <= 19.5: return 0.6833
    else: return 0.25
  else: return 0.678

==== end of DT for dataset `liver-disorders'. F1 score: 0.774193548387 ====
```