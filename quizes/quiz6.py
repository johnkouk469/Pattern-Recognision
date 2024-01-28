import pandas as pd
data = pd.read_csv("C:/Users/johnk/Documents/Google Drive/Pattern-Recognision/quiz6_data.csv", sep=",")

trainingRange = list(range(0, 50)) + list(range(90,146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, "Type"]
training = training.drop(["Type"], axis=1)

testingRange = list(range(50, 90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:, "Type"]
testing = testing.drop("Type", axis=1)

# training = training.drop_duplicates()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(training), columns=training.columns)
scaler = scaler.fit(testing)
transformedTesting = pd.DataFrame(scaler.transform(testing), columns=testing.columns)
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

import matplotlib.pyplot as plt
# plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
# plt.show()


# question 1
print(eigenvalues[0]/sum(eigenvalues))

pca = PCA(n_components=4)
pca = pca.fit(transformed)
pca_transformed = pd.DataFrame(pca.transform(transformed))
# plt.scatter(pca_transformed.loc[:, 0], pca_transformed.loc[:, 1])
# plt.show()

pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=training.columns)

# question 2
info_loss = 1 - (eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + eigenvalues[3])/sum(eigenvalues)  
print(info_loss)

# question 3
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(training, trainingType)
pred = clf.predict(testing)

from sklearn.metrics import accuracy_score, recall_score
print(accuracy_score(testingType, pred))
# question 4
print(recall_score(testingType,pred))

# question 5
import numpy as np
testing_error = [ 0 for i in range(len(eigenvalues)+1)]
for i in range(1,len(eigenvalues)+1):
    pca = PCA(n_components=i)
    pca = pca.fit(transformed)
    pca_transformed = pd.DataFrame(pca.transform(transformed))
    pca_inverse = pd.DataFrame(pca.inverse_transform(pca_transformed), columns=training.columns)
    clf = KNeighborsClassifier(n_neighbors=3).fit(pca_inverse,trainingType)
    pca = pca.fit(transformedTesting)
    pca_transformed_testing = pca.transform(transformedTesting)
    pca_inverse_testing = pd.DataFrame(pca.inverse_transform(pca_transformed_testing), columns=testing.columns)
    pred = clf.predict(pca_inverse_testing)
    testing_error[i] = accuracy_score(testingType, pred)
    
print(testing_error)