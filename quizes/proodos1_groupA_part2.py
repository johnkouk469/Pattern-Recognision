# Εισαγωγή των δεδομένων
import pandas as pd
covid_data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/covid-data.csv")

# χωρίζω training set και test set
import numpy as np
from sklearn.impute import SimpleImputer
X = pd.DataFrame(covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"],
                        covid_data.loc[:, [np.nan, "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]]],
                        covid_data.loc[:, ["Fever", np.nan, "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", np.nan, "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", np.nan, "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", np.nan, "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", np.nan, "Diarrhea", "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", np.nan, "NoneSymptoms", "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", np.nan, "Age", "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", np.nan, "Gender"]],
                        covid_data.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", np.nan]], dtype="category")
                        imp = SimpleImputer(strategy="most_frequent")
                        imp.fit_transform(X)
Y = pd.DataFrame(covid_data.loc[:, "Severity"], covid_data.loc[:, np.nan], dtype="category")

xtrain = X.loc[1:9000, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender"]]
xtest = 
ytest = covid_data.loc[9001:10001, "Severity", np.nan]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder = encoder.fit(xtrain)
xtrain = encoder.transform(xtrain)
# encoder = encoder.fit(ytrain)
# ytrain = encoder.transform(ytrain)
encoder = encoder.fit(xtest)
xtest = encoder.transform(xtest)
# encoder = encoder.fit(ytest)
# ytest = encoder.transform(ytest)

from sklearn.naive_bayes import GaussianNB
import numpy as np
clf = GaussianNB()
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
predprob = clf.predict_proba(xtest)
print(pred)
print(predprob)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

print("Accuracy: ", accuracy_score(ytest, pred))
print("Precision: ", precision_score(ytest, pred, pos_label=1))
print("Recall: ", recall_score(ytest, pred, pos_label=1))
print("F1 Score: ", f1_score(ytest, pred, pos_label=1))

fpr, tpr, thresholds = roc_curve(ytest, predprob[:, 1])
print("AUC: ", auc(fpr, tpr))

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc(fpr, tpr))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

