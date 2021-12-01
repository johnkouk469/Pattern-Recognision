import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]

Y = [1, 1, 1, 1, 2, 2, 2, 2]

alldata = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
X = alldata.loc[:, ["X1", "X2"]]
y = alldata.Y

plt.scatter(X[(y == 1)].X1, X[(y==1)].X2, c="red", marker="+")
plt.scatter(X[(y == 2)].X1, X[(y==2)].X2, c="blue", marker="o")
# plt.show()

x1 = np.arange(min(X.X1.tolist()), max(X.X1.tolist()), 0.01)
x2 = np.arange(min(X.X2.tolist()), max(X.X2.tolist()), 0.01)
xx, yy = np.meshgrid(x1, x2)

from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC(kernel="rbf", gamma=1)
clf = clf.fit(X, y)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="cyan")
# plt.show()

clf = svm.SVC(kernel="rbf", gamma=1)
clf = clf.fit(X, y)
pred = clf.predict(X)
print(accuracy_score(y, pred))

clf = svm.SVC(kernel="rbf", gamma=1000000)
clf = clf.fit(X, y)
pred = clf.predict([[-2.0, -1.9]])
print(pred)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="purple")
plt.show()