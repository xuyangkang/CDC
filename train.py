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
    row = 916750
    col = 300
    util = utils.Utils(vecs, row, col, words)
    df = pd.read_csv(train)
    train_size = df.shape[0] // 10 * 9
    test_size = df.shape[0] - train_size
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for index, row in df.iterrows():
        x = util.embed_sentense(row['text'])
        x = np.append(x, [int(row['sex']) - 1, float(row['age']) / 100])
        y = np.array([int(row['event'])])
        if index < train_size:
            train_x.append(x)
            train_y.append(y)
        else:
            test_x.append(x)
            test_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    train_y = keras.utils.to_categorical(train_y, 100)
    test_y = keras.utils.to_categorical(test_y, 100)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    model = Sequential()
    model.add(Dense(units=400, activation='relu', input_shape=(302,)))
    model.add(Dense(units=400, activation='relu'))
    model.add(Dense(units=100, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=30, batch_size=20)
    score = model.evaluate(test_x, test_y)

    print(score)


if __name__ == '__main__':
    train_model()
