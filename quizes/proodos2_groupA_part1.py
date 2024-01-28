import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/data.csv")

X = data.loc[:, ["Exports", "Health", "Income", "Age"]]
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler = scaler.fit(X)
# transformed = pd.DataFrame(scaler.transform(X), columns=["Exports", "Health", "Income", "Age"])

# from sklearn.decomposition import PCA
# pca = PCA()
# pca = pca.fit(transformed)
# pca_transformed = pd.DataFrame(pca.transform(transformed))
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_

# info_loss = 1 - (eigenvalues[0] + eigenvalues[1] + eigenvalues[2])/sum(eigenvalues)  
# print(info_loss)

set9 = data.loc[:, ["ChildMortality", "Health", "Age"]]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init=set9.loc[0:1, :]).fit(set9)
print(kmeans.cluster_centers_)
# print(kmeans.labels_)
# print(kmeans.inertia_)


# from numpy.core.fromnumeric import mean
# from sklearn.metrics import silhouette_samples, silhouette_score
# silhouette_scores = []
# set10 = data.loc[:, ["Exports", "Imports", "Age"]]
# for n in range(10,20):
#     kmeans = KMeans(n_clusters=n, init=set10.loc[0:n-1, :]).fit(set10)
#     silhouette_scores.append(silhouette_score(set10, kmeans.labels_))
#     # for i in range(n):
#     #     print(mean(silhouette_samples(set10, kmeans.labels_)[kmeans.labels_ == i]))
# print(silhouette_scores)    


# from sklearn.cluster import AgglomerativeClustering
# clustering = AgglomerativeClustering(n_clusters=5, linkage="single").fit(set9)
# print(silhouette_score(set9, clustering.labels_))


# set12 = data.loc[:, ["ChildMortality", "Exports", "Income"]]

# from sklearn.cluster import DBSCAN
# silhouette_scores = []
# for min_samples in range(5,11):
#     clustering = DBSCAN(eps=10, min_samples=min_samples).fit(set12)
#     silhouette_scores.append(silhouette_score(set12, clustering.labels_))
# print(silhouette_scores)