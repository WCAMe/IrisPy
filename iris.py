import numpy as np
import sklearn
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.simplefilter(action = 'ignore', category = FutureWarning)

iris = load_iris()
x = iris['data']
y = iris['target']

enc = OneHotEncoder()
y = enc.fit_transform(y[:, np.newaxis]).toarray()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.5, random_state = 2)

n_features = x.shape[1]
n_classes = y.shape[1]

model = keras.Sequential(name = 'Iris')
model.add(keras.layers.Dense(8, input_dim = n_features, activation = 'relu'))
model.add(keras.layers.Dense(n_classes, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=5, epochs=50, verbose=0, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])