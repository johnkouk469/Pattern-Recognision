import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np


X1 = [2,2,-2,-2,1,1,-1,-1]
X2 = [2,-2,-2,2,1,-1,-1,1]
Y = [1,1,1,1,2,2,2,2]
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})
xtrain = alldata.loc[:, ["X1", "X2"]]
ytrain = alldata.loc[:,"Y"]


model1 = MLPRegressor(hidden_layer_sizes=(2,),max_iter=10000).fit(xtrain,ytrain)
pred = model1.predict(xtrain)
trainingError = [ (t-p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)
# plt.hist(trainingError, range=(-1, 1), rwidth=0.5)
# plt.show()

model2 = MLPRegressor(hidden_layer_sizes=(20,),max_iter=10000).fit(xtrain,ytrain)
pred = model2.predict(xtrain)
trainingError = [ (t-p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)

model3 = MLPRegressor(hidden_layer_sizes=(20,20),max_iter=10000).fit(xtrain,ytrain)
pred = model3.predict(xtrain)
trainingError = [ (t-p) for (t, p) in zip(ytrain, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)