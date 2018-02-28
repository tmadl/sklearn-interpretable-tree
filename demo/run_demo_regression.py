from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from InterpretableDecisionTreeRegression import IDecisionTreeRegressor

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

predicted1 = cross_val_predict(lr, boston.data, y, cv=10)
predicted2 = cross_val_predict(IDecisionTreeRegressor(max_depth=5), boston.data, y, cv=10)

print(IDecisionTreeRegressor(max_depth=5).fit(boston.data, y))

fig, ax = plt.subplots()
ax.scatter(y, predicted1, color='b')
ax.scatter(y, predicted2, color='r')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()