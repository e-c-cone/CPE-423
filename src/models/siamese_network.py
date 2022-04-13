import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input


class NeuralNetwork:
    def __init__(self, name: str, **kwargs):
        self.__dict__.update(kwargs)
        self.model = tf.keras.models.Sequential([
            Flatten(input_shape=self.inp_shape),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(512, activation='relu'),
            Dense(self.out_shape)
        ])
        self.name = name
        self.model.compile(self.optimizer, self.loss, self.metrics)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=15)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]

    def predict(self, x):
        return self.model.predict(x)import tensorflow as tf