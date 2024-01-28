import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/quiz9_data.csv")
target = data.loc[:, "Y"]
data = data.drop(["Y"], axis=1)

plt.scatter(data.X1, data.X2)
plt.show()

plt.scatter(data.X1, data.X2, c=target, cmap="bwr")
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(data)
plt.scatter(data.X1, data.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, tol=0.0001).fit(data)
print(gm.means_)
print(gm.covariances_)
print(np.sum(gm.score_samples(data)))

x = np.linspace(np.min(data.loc[:, "X1"]), np.max(data.loc[:, "X1"]))
y = np.linspace(np.min(data.loc[:, "X2"]), np.max(data.loc[:, "X2"]))
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z)
plt.scatter(data[(target == 1)].X1, data[(target == 1)].X2, c="red", marker="+")
plt.scatter(data[(target == 2)].X1, data[(target == 2)].X2, c="green", marker="o")
plt.scatter(data[(target == 3)].X1, data[(target == 3)].X2, c="blue", marker="x")
plt.show()
