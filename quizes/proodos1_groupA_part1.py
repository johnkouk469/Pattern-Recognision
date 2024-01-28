import pandas as pd
import matplotlib.pyplot as plt

training = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/training.csv")
testing = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/testing.csv")

absfreq = pd.crosstab(training.Time, training.Severity)
freq = pd.crosstab(training.Time, training.Severity, normalize='index')
freqSum = pd.crosstab(training.Time, training.Severity, normalize='all').sum(axis=1)
GINI_Day = 1 - freq.loc["Day", "Low"]**2 - freq.loc["Day", "High"]**2
GINI_Night = 1 - freq.loc["Night", "Low"]**2 - freq.loc["Night", "High"]**2
GINI_Time = freqSum.loc["Day"] * GINI_Day + freqSum["Night"] * GINI_Night
print(round(GINI_Time,4))


from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB

X = training.loc[:, ["Temperature", "Visibility", "Pressure"]]
y = training.loc[:, "Severity"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder = encoder.fit(X)
X = encoder.transform(X)

clf = CategoricalNB(alpha=1)
clf.fit(X, y)

new_data = pd.DataFrame({"Temperature": ["71.0"], "Visibility": ["9.0"], "Pressure": ["31.0"]})
transformed_new_data = encoder.transform(new_data)
print(clf.predict(transformed_new_data))
print(clf.predict_proba(transformed_new_data))

xtrain = training.loc[:, ["Temperature", "Humidity", "Visibility"]]
ytrain = training.loc[:, "NumSeverity"]

from sklearn.neural_network import MLPRegressor
import numpy as np

model = MLPRegressor(hidden_layer_sizes=(20,20),max_iter=10000).fit(xtrain,ytrain)
pred = model.predict(xtrain)
trainingError = [ (t-p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
xtrain = training.loc[:, ["Temperature", "Humidity", "Pressure"]]
ytrain = training.loc[:, "NumSeverity"]
xtest = testing.loc[:, ["Temperature", "Humidity", "Pressure"]]
ytest = testing.loc[:, "NumSeverity"]
Kvalues = [5, 6, 7, 8, 9, 10]
recalles = []
for k in Kvalues:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf = clf.fit(xtrain, ytrain)
    pred = clf.predict(xtest)
    recalles.append(recall_score(ytest, pred, pos_label=1))
    print("Recall: ", recall_score(ytest, pred, pos_label=1))
    
# print("Best k: ", Kvalues[np.argmax(recalles)])

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
xtrain = training.loc[:, ["Wind", "Humidity", "Visibility"]]
ytrain = training.loc[:, "NumSeverity"]
gammavalues = [1, 5, 12, 15]
accuracies = []
for gamma in gammavalues:
    clf = svm.SVC(kernel="rbf", gamma=gamma)
    scores = cross_val_score(clf, xtrain, ytrain, cv=10)
    accuracies.append(scores.mean())

print("Best gamma: ", gammavalues[np.argmax(accuracies)])