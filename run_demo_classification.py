import matplotlib.pyplot as plt, numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from InterpretableDecisionTreeClassifier import IDecisionTreeClassifier
from treeutils import tree_to_code
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import f1_score

X, y = make_moons(300, noise=0.4)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

clf1 = DecisionTreeClassifier(max_depth=4).fit(Xtrain,ytrain)
clf2 = IDecisionTreeClassifier(max_depth=4).fit(Xtrain,ytrain)

print("=== original decision tree ===")
features = ["ft"+str(i) for i in range(X.shape[1])]
print(tree_to_code(clf1, features)) # output large tree
print("=== simplified (interpretable) decision tree ===")
print(tree_to_code(clf2, features))

h = 0.02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


plt.subplot(1,2,1)
plt.title("original decision tree. F1: "+str(f1_score(ytest, clf1.predict(Xtest))))
Z = clf1.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
plt.scatter(X[:,0], X[:,1], c=y)

plt.subplot(1,2,2)
plt.title("simplified (interpretable) decision tree. F1: "+str(f1_score(ytest, clf2.predict(Xtest))))
Z = clf2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()