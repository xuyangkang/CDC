import numpy as np
import pandas as pd
import utils
import click
import tensorflow.keras as keras


@click.command()
@click.option('--train', default='./train.csv', help='The given train file')
@click.option('--vecs', default='./word_vec.npy', help='The word vector mmap file')
@click.option('--words', default='./words.txt', help='The word list')
def convert_train_data(train, vecs, words):
    row = 916750
    col = 300
    util = utils.Utils(vecs, row, col, words)
    df = pd.read_csv(train)
    data_x = []
    data_y = []

    for index, row in df.iterrows():
        x = util.embed_sentense(row['text'])
        x = np.append(x, [int(row['sex']) - 1, float(row['age']) / 100])
        y = np.array([int(row['event'])])
        data_x.append(x)
        data_y.append(y)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_y = keras.utils.to_categorical(data_y, 100)

    np.save('data_x.npy', data_x)
    np.save('data_y.npy', data_y)


if __name__ == '__main__':
    # pylint: disable=locally-disabled, no-value-for-parameter
    convert_train_data()
