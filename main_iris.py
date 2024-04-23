import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# def sigmoid_activation(X):
#     return 1/(1+np.exp(-X))
#
#
# def sigmoid_derivative(X):
#     return X*(1-X)


def step_function(X):
    X[X>0] = 1
    X[X<=0] = 0
    return X


def predict(X, W):
    return np.dot(X,W)



data = pd.read_csv('iris.csv')

X = data.drop('Id', axis = 1)

X = X.iloc[:100, :-1]
y = data.iloc[:100, -1]
y = np.where(y == 'Iris-setosa', 1, 0)
bias_col = np.ones((X.shape[0],1))

X = np.hstack((X,np.ones((X.shape[0],1))))

X_train, X_test, y_train, y_test = train_test_split(X, y)

epochs = 50
alpha = 0.01

W = np.random.randn(X.shape[1])

losses = []
for epoch in range(epochs):
    preds = step_function(predict(X_train, W))
    errors = preds - y_train
    loss = np.sum(errors **2)
    losses.append(loss)

    W += -alpha * np.dot(X_train.T, errors)

    # for example, label in zip(X_train, y_train):
    #     p = step_function(predict(example, W))
    #     if p != label:
    #         error = p - label
    #         loss = error ** 2
    #         losses.append(loss)
    #         W += -alpha * error * example

p = step_function(predict(X_test,W))
print('error sum:', np.sum(p-y_test))
print(losses)