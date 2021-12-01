import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/quizes/quiz2_data.csv")
# print(data)

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# encoder.fit(data.loc[:, ['CustomerID']])
# transformedCustomerID = encoder.transform(data.loc[:, ['CustomerID']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedCustomerID, data.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()

# encoder.fit(data.loc[:, ['Sex']])
# transformedSex = encoder.transform(data.loc[:, ['Sex']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedSex, data.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()

# encoder.fit(data.loc[:, ['CarType']])
# transformedCarType = encoder.transform(data.loc[:, ['CarType']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedCarType, data.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()

# encoder.fit(data.loc[:, ['Budget']])
# transformedBudget = encoder.transform(data.loc[:, ['Budget']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedBudget, data.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()

absfreq = pd.crosstab(data.Sex, data.Insurance)
freq = pd.crosstab(data.Sex, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Sex, data.Insurance, normalize='all').sum(axis=1)
GINI_M = 1 - freq.loc["M", "No"]**2 - freq.loc["M", "Yes"]**2
GINI_F = 1 - freq.loc["F", "No"]**2 - freq.loc["F", "Yes"]**2
GINI_Sex = freqSum.loc["M"] * GINI_M + freqSum["F"] * GINI_F
print(GINI_Sex)

absfreq = pd.crosstab(data.CarType, data.Insurance)
freq = pd.crosstab(data.CarType, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.CarType, data.Insurance, normalize='all').sum(axis=1)
GINI_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
GINI_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
GINI_Sedan = 1 - freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
GINI_CarType = freqSum.loc["Family"] * GINI_Family + freqSum["Sport"] * GINI_Sport + freqSum.loc["Sedan"] * GINI_Sedan
print(GINI_CarType)

absfreq = pd.crosstab(data.Budget, data.Insurance)
freq = pd.crosstab(data.Budget, data.Insurance, normalize='index')
freqSum = pd.crosstab(data.Budget, data.Insurance, normalize='all').sum(axis=1)
GINI_Low = 1 - freq.loc["Low", "No"]**2 - freq.loc["Low", "Yes"]**2
GINI_Medium = 1 - freq.loc["Medium", "No"]**2 - freq.loc["Medium", "Yes"]**2
GINI_High = 1 - freq.loc["High", "No"]**2 - freq.loc["High", "Yes"]**2
GINI_VeryHigh = 1 - freq.loc["VeryHigh", "No"]**2 - freq.loc["VeryHigh", "Yes"]**2
GINI_Budget = freqSum.loc["Low"] * GINI_Low + freqSum["Medium"] * GINI_Medium + freqSum.loc["High"] * GINI_High + freqSum["VeryHigh"] * GINI_VeryHigh
print(GINI_Budget)