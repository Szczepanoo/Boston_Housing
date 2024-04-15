import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Wczytanie danych
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', 'BIAS_COL']

df = pd.read_csv('hou_all.csv', header=None, names=column_names)

X = df.drop(columns=['MEDV', 'BIAS_COL'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Podział danych na cechy (X) i etykietę (y)
X = df.drop(columns=['MEDV', 'BIAS_COL'])
y = df['MEDV']

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='mean_squared_error')

# Trenowanie modelu
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Ocena modelu na danych testowych
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

# Wykresy przebiegu nauki
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='center right')
plt.show()
