from os import sep
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

csv_data = pd.read_csv("C:/Users/johnk/Documents/Google Drive/Pattern-Recognision/quiz7_data.csv")
data = csv_data.loc[:, ["X1", "X2"]]

init_X1 = [-4, 0, 4]
init_X2 = [10, 0, 10]
# labels = ["center1", "center2", "center3"]
init_data = pd.DataFrame({"X1": init_X1, "X2": init_X2})
# print(init_data.loc[:, :])
# print(data)
kmeans = KMeans(n_clusters=3, init=init_data).fit(data)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(kmeans.inertia_)
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = data.mean()
print(m)
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)    
print(separation)
print(Ci)

plt.scatter(data.X1, data.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score
print(silhouette_score(data, kmeans.labels_))
print(mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 0]))
print(mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 1]))
print(mean(silhouette_samples(data, kmeans.labels_)[kmeans.labels_ == 2]))


init_X1 = [-2, 2, 0]
init_X2 = [0, 0, 10]
init_data = pd.DataFrame({"X1": init_X1, "X2": init_X2})
kmeans2 = KMeans(n_clusters=3, init=init_data).fit(data)
print(kmeans2.cluster_centers_)
print(kmeans2.labels_)
print(kmeans2.inertia_)
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = data.mean()
print(m)
for i in list(set(kmeans2.labels_)):
    mi = data.loc[kmeans2.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans2.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)    
print(separation)
print(Ci)

plt.scatter(data.X1, data.X2, c=kmeans2.labels_)
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print(silhouette_score(data, kmeans2.labels_))
print(mean(silhouette_samples(data, kmeans2.labels_)[kmeans2.labels_ == 0]))
print(mean(silhouette_samples(data, kmeans2.labels_)[kmeans2.labels_ == 1]))
print(mean(silhouette_samples(data, kmeans2.labels_)[kmeans2.labels_ == 2]))
