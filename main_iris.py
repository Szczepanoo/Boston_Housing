import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid_activation(X):
    return 1/(1+np.exp(-X))
def sigmoid_derivative(X):
    return X*(1-X)
def predict(X, W):
    return sigmoid_activation(np.dot(X, W))

data = pd.read_csv('iris.csv')

X = data.drop('Id', axis = 1)

X = X.iloc[:100, :-1]
y = data.iloc[:100, -1]
y = np.where(y == 'Iris-setosa', 1, 0)
bias_col = np.ones((X.shape[0],1))

X = np.hstack((X,bias_col))

X_train, X_test, y_train, y_test = train_test_split(X, y)

epochs = 50
alpha = 0.01

W = np.random.randn(X.shape[1])

losses = []
for epoch in range(epochs):
    predictions = predict(X_train, W)
    error = predictions - y_train
    loss = np.sum(error**2)
    losses.append(loss)

    d = error * sigmoid_derivative(predictions)
    gradient = np.dot(X_train.T , d)

    W += -alpha * gradient
print(losses)