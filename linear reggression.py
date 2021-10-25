import sklearn.datasets as datasets
iris = datasets.load_iris()
import sklearn.datasets as datasets

print("iris dataset is {}".format(iris.DESCR))
print("iris data size is {}".format(iris.data.shape))
print("iris target size is {}".format(iris.target.shape))
print("iris data has {} features, the feature names are {}".format(
    iris.data.shape[1], iris.feature_names))
print("iris data has {} samples, the target label names {}".format(
    iris.data.shape[1], iris.target_names))
import sklearn.preprocessing as preprocessing
import numpy as np

X = np.random.randint(2, 10, size=(4, 2))
X2 = np.random.randint(100, 10000, size=(4, 2))
X = np.concatenate((X, X2), axis=1)
print("The original matrix")
print(X)

#### min-max scaler

minmax = preprocessing.MinMaxScaler()
minmax.fit(X)
X_minmax = minmax.transform(X)
print("The transform data using min-max scaler")
import sklearn.preprocessing as preprocessing
import numpy as np

X = np.random.randint(2, 10, size=(4, 2))
X2 = np.random.randint(100, 10000, size=(4, 2))
X = np.concatenate((X, X2), axis=1)
print("The original matrix")
print(X)

std = preprocessing.StandardScaler()
std.fit(X)
X_std = std.transform(X)
print("The transform data using Standard scaler")
print(X_std)
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

house = datasets.load_boston()

print("The data shape of house is {}".format(house.data.shape))
print("The number of feature in this data set is {}".format(
    house.data.shape[1]))

train_x, test_x, train_y, test_y = train_test_split(house.data,
                                                    house.target,
                                                    test_size=0.2,
                                                    random_state=42)

print("The first five samples {}".format(train_x[:5]))
print("The first five targets {}".format(train_y[:5]))
print("The number of samples in train set is {}".format(train_x.shape[0]))
print("The number of samples in test set is {}".format(test_x.shape[0]))

lr = LinearRegression()
lr.fit(train_x, train_y)

pred_y = lr.predict(test_x)
print("The first five prediction {}".format(pred_y[:5]))
print("The real first five labels {}".format(test_y[:5]))

mse = metrics.mean_squared_error(test_y, pred_y)
print("Mean Squared Error {}".format(mse))
print(X_minmax)
