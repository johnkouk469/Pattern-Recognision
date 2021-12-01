from sklearn import datasets
import statistics

iris = datasets.load_iris()
petalLengthIndex = iris.feature_names.index('petal length (cm)')
petalWidthIndex = iris.feature_names.index('petal width (cm)')
sepalLengthIndex = iris.feature_names.index('sepal length (cm)')
sepalWidthIndex = iris.feature_names.index('sepal width (cm)')
data = iris.data

meanPetalLength = round((statistics.mean(data[:,petalLengthIndex])),2)
print(meanPetalLength)

maxOfSepalWidth = round(max(data[:,sepalWidthIndex]),1)
print(maxOfSepalWidth)

varianceSepalLength = round(statistics.variance(data[:,sepalLengthIndex]),2)
print(varianceSepalLength)

meanPetalWidth = round((statistics.mean(data[:,petalWidthIndex])),2)
meanSepalLength = round((statistics.mean(data[:,sepalLengthIndex])),2)
meanSepalWidth = round((statistics.mean(data[:,sepalWidthIndex])),2)
meanCol = [meanPetalLength, meanPetalWidth, meanSepalLength, meanSepalWidth]
print(meanCol)