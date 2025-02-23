import numpy as np
import pandas as pd
import utils
import click
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


@click.command()
@click.option('--train', default='./train.csv', help='The given train file')
@click.option('--vecs', default='./word_vec.npy', help='The word vector mmap file')
@click.option('--words', default='./words.txt', help='The word list')
def train_model(train, vecs, words):
    data_x = np.load('data_x.npy')
    data_y = np.load('data_y.npy')

    model = Sequential()
    model.add(Dense(units=200, activation='relu', input_shape=(302,)))
    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=100, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # model.fit(data_x, data_y, epochs=30, batch_size=20, validation_split=0.1)
    model.fit(data_x, data_y, epochs=30, batch_size=20)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    # pylint: disable=locally-disabled, no-value-for-parameter
    train_model()
