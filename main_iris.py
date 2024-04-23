import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self,N, epochs = 10, alpha = 0.1):
        self.epochs = epochs
        self.alpha = alpha
        self.W = np.random.randn(N)

    def activation(self, X):
        X = np.where(X>=0,1,0)
        return X
    def fit(self, X, y):
        losses = []
        for epoch in range(self.epochs):
            p = self.predict(X)
            errors = p - y
            loss = np.sum(errors ** 2) / 2
            losses.append(loss)
            self.W += -self.alpha * np.dot(X.T, errors)

        print(f'losses: {losses}')

    def predict(self, X):
        return self.activation(np.dot(X,self.W))




data = pd.read_csv('iris.csv')

X = data.iloc[:100, 1:-1]
y = data.iloc[:100, -1]
y = np.where(y == 'Iris-setosa', 1, 0)

X= np.hstack((X, np.ones((X.shape[0], 1))))

X_train, X_test, y_train, y_test = train_test_split(X, y)
P = Perceptron(X.shape[1],epochs=20)
P.fit(X_train, y_train)

pred = y_test - P.predict(X_test)
print(np.sum(y_test - P.predict(X_test)))
