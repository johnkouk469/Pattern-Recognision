import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

weather = pd.read_csv("./weather.txt")
print(weather)
