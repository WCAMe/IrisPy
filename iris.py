import numpy as np
import sklearn
import warnings
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
warnings.simplefilter(action = 'ignore', category = FutureWarning)
plt.style.use('ggplot')
%matplotlib inline

iris = load_iris()
x = iris['data']
y = iris['target']
names = iris['target_names']
feature_name = iris['feature_names']

enc = OneHotEncoder()
y = enc.fit_transform(y[:, np.newaxis]).toarray()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.5, random_state = 2)

n_features = x.shape[1]
n_classes = y.shape[1]

def create_custom_model(input_dim, output_dim, nodes, n = 1, name = 'model'):
    def create_model():
        model = keras.Sequential(name = name)
        for i in range(n):
            model.add(keras.layers.Dense(nodes, input_dim = input_dim, activation = 'relu'))
        model.add(keras.layers.Dense(output_dim, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 8, i, 'model{}'.format(i)) for i in range(1, 4)]

history_dict = {}

for create_model in models:
    model = create_model()
    print('Model name:', model.name)
    history_callback = model.fit(x_train, y_train,
                                 batch_size=5,
                                 epochs=50,
                                 verbose=0,
                                 validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    history_dict[model.name] = [history_callback, model]

create_model = create_custom_model(n_features, n_classes, 8, 3)

estimator = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
scores = cross_val_score(estimator, x_scaled, y, cv=10)
print("Accuracy : {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std()))