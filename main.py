import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def sigmoid_activation(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X):
    return X * (1-X)

def predict(X,W):
    return sigmoid_activation(np.dot(X,W))

data = pd.read_csv('hou_all.csv', header=None, names=('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                                                      'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                                      'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL'))


X = data.drop('MEDV', axis=1)
y = data.iloc[:, 13]
epochs = 100
learning_rate = 1
# print(X)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
W = np.random.randn(X.shape[1])
# print(W)

losses = []
for epoch in range(epochs):
    prediction = predict(X_train,W)
    # print(prediction)
    errors = y_train - prediction
    losses.append(errors.sum())

    W += -learning_rate * X_train.T.dot(errors)

print(losses)