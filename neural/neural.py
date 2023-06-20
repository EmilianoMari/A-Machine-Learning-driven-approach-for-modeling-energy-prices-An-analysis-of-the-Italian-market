from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os


def best_prediction(array):
    m = array[0]
    index = 0
    for i in range(len(array)):
        if array[i] > m:
            m = array[i]
            index = i
        else:
            continue
    return index, array[index]


class Neural(object):
    def __init__(self, market, model_type, scope, metrics, dataset_len, epochs=None, batch=None):
        self.market = market
        self.metrics = metrics
        self.n_metrics = len(metrics)
        self.dataset_len = dataset_len
        self.epochs = epochs
        self.batch = batch
        self.scope = scope
        self.model = None
        self.model_type = model_type
        self.path = "bi-data/%s/model/%s_" % (self.market, self.dataset_len)

        self.model_inventory = {
            1: self.model_1,
            2: self.model_2,
            3: self.model_3
        }

    def model_1(self, x_train):
        embedding_dim = 10 * self.dataset_len
        input_layer = layers.Input(shape=(x_train.shape[1],))
        x = layers.Embedding(15000, embedding_dim, input_length=self.n_metrics * 10)(input_layer)
        x = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(256, dropout=0.2, return_sequences=True)(x)
        x = layers.LSTM(128, dropout=0.1)(x)
        return input_layer, x

    def model_2(self, x_train):
        embedding_dim = 10 * self.dataset_len
        input_layer = layers.Input(shape=(x_train.shape[1],))
        x = layers.Embedding(15000, embedding_dim, input_length=self.n_metrics * 10)(input_layer)
        x = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(250, dropout=0.2)(x)
        return input_layer, x

    def model_3(self, x_train):
        embedding_dim = 10 * self.dataset_len
        input_layer = layers.Input(shape=(x_train.shape[1],))
        x = layers.Embedding(15000, embedding_dim, input_length=self.n_metrics * 10)(input_layer)
        x = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(256, dropout=0.2, return_sequences=True)(x)
        x = layers.LSTM(128, dropout=0.1)(x)
        return input_layer, x

    def create(self, x_train, y_train, x_test=None, y_test=None):
        input_layer, x = self.model_inventory[self.model_type](x_train)

        if self.n_metrics == 1:
            out = layers.Dense(max(y_train) + 2, activation="softmax")(x)
            self.model = Model(inputs=input_layer, outputs=out)
        else:
            output = []
            for i in range(self.n_metrics):
                output.append(layers.Dense(max(y_train[i]) + 2, activation="softmax")(x))
            self.model = Model(inputs=input_layer, outputs=output)

        self.model.summary()
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        result = self.model.fit(x_train, y_train, batch_size=self.batch, epochs=self.epochs)
        if x_test is not None and y_test is not None:
            self.model.evaluate(x_test, y_test)
        self.save_model(result)

    def resume_train(self, x_train, y_train, epochs):
        self.load()
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        result = self.model.fit(x_train, y_train, batch_size=self.batch, epochs=epochs)
        self.save_model(result)

    def save_model(self, result):
        with open(self.path + "%s.json" % self.scope, "w+") as file:
            file.write(self.model.to_json())
        self.model.save_weights(self.path + "%s.h5" % self.scope)

        with open(self.path + "%s_result.json" % self.scope, "w+") as file:
            result = pd.DataFrame(result.history)
            result.to_json(file)

    def load(self):
        with open(self.path + "%s.json" % self.scope, "r") as file:
            loaded_model_json = file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.path + "%s.h5" % self.scope)
        self.model.summary()

    def predict(self, array, length_array=None):
        if length_array is None:
            length_array = self.n_metrics * self.dataset_len

        num_array = np.asarray(array)
        num_array = num_array.reshape((1, length_array))
        array_res = self.model.predict(num_array, verbose=0)

        # return array_res

        if self.n_metrics == 1:
            array_res = [array_res]

        res_a = []
        for res in array_res:
            res_a.append(best_prediction(res[0]))
        return res_a

    def model_exist(self):
        return os.path.isfile(self.path + "%s.h5" % self.scope)
