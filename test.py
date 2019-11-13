import numpy as np
import pandas as pd
import utils
import click
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json


@click.command()
@click.option('--test', default='./test.csv', help='The given test file')
@click.option('--predict', default='./solution.csv', help='The solution')
@click.option('--vecs', default='./word_vec.npy', help='The word vector mmap file')
@click.option('--words', default='./words.txt', help='The word list')
def predict(test, predict, vecs, words):
    row = 916750
    col = 300
    util = utils.Utils(vecs, row, col, words)

    df = pd.read_csv(test)
    data_x = []

    for index, row in df.iterrows():
        x = util.embed_sentense(row['text'])
        x = np.append(x, [int(row['sex']) - 1, float(row['age']) / 100])
        data_x.append(x)

    data_x = np.array(data_x)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    data_y = model.predict(data_x)
    data_y = np.argmax(data_y, axis=1)
    df['event'] = pd.Series(data_y)
    df.to_csv(predict, index=False)


if __name__ == '__main__':
    # pylint: disable=locally-disabled, no-value-for-parameter
    predict()
