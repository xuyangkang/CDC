import numpy as np
import logging

class Utils(object):
    def __init__(self, mmf, row, column, words):
        self._fp = np.memmap(mmf, dtype='float16', mode='r', shape=(row, column))
        self._d = column
        self._words_to_id = {}
        self._words = {}
        with open(words) as f:
            id = 0
            for line in f:
                word = line.strip()
                self._words[id] = word
                self._words_to_id[word] = id
                id += 1

    def embed_sentense(self, s : str) -> np.array:
        tot = 0
        acc = np.zeros(self._d)
        for word in s.split():
            if not word:
                continue
            id = self._words_to_id.get(word, -1)
            if id != -1:
                acc += np.array(self._fp[id])
                tot += 1
            else:
                # logging.warning(f'unknown word: {word}')
                pass

        if tot > 0:
            return acc / tot
        else:
            return acc

