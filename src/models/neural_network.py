import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input


class NeuralNetwork:
    def __init__(self, name: str, **kwargs):
        self.__dict__.update(kwargs)
        self.model = tf.keras.models.Sequential([
            Flatten(input_shape=self.inp_shape),
            Dense(512, activation='relu'),
            Dense(688, activation='relu'),
            Dropout(0.3),
            Dense(512, activation='relu'),
            Dense(self.out_shape)
        ])
        self.name = name
        self.model.compile(self.optimizer, self.loss, self.metrics)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=self.epochs)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        self.model.save(os.path.join(self.dir, self.fname))

    def load(self):
        self.model = tf.keras.models.load_model(os.path.join(self.dir, self.fname))