import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

data = pd.read_csv("C:/Users/John/Documents/Google Drive/Pattern-Recognision/quizes/quiz3_data.csv")
print(data)
ytest = data.loc[:,'Class']
predprobM1 = data.loc[:,'P_M1']
predprobM2 = data.loc[:,'P_M2']
fpr1, tpr1, thresholds = roc_curve(ytest, predprobM1)
fpr2, tpr2, thresholds = roc_curve(ytest, predprobM2)

for i in range(len(tpr1)):
    print("tpr1 is ", tpr1[i], "and threshold is ", thresholds[i])
 
print("F1 Score: ", round(f1_score(ytest, round(predprobM2), pos_label=1),4))

print("AUC_M1: ", auc(fpr1, tpr1))
print("AUC_M2: ", auc(fpr2, tpr2))

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc(fpr1, tpr1))
plt.plot(fpr2, tpr2, 'g', label = 'AUC = %0.2f' % auc(fpr2, tpr2))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()