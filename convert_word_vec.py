import numpy as np
import click
import re


@click.command()
@click.option('--input', default='./crawl-300d-2M.vec', help='The extracted word vectors')
@click.option('--vecs', default='./word_vec.npy', help='The output word vector mmap file')
@click.option('--words', default='./words.txt', help='The word list')
def convert_word_vec(input, vecs, words):
    r = 0
    c = 0
    seen_words = set()
    with open(input) as in_f:
        for line in in_f:
            pieces = line.strip().split()
            if len(pieces) == 2:
                r = int(pieces[0])
                c = int(pieces[1])
            else:
                word = pieces[0].upper()
                if not re.match(r'^[A-Z]+$', word):
                    continue
                if word in seen_words:
                    continue
                seen_words.add(word)

    r = len(seen_words)
    written_words = set()
    count = 0
    with open(input) as in_f:
        with open(words, 'w+') as out_f:
            for line in in_f:
                pieces = line.strip().split()
                if len(pieces) == 2:
                    mmap_file = np.memmap(vecs, dtype='float16', mode='w+', shape=(r,c))
                else:
                    word = pieces[0].upper()
                    if not word in seen_words:
                        continue
                    if word in written_words:
                        continue
                    written_words.add(word)
                    out_f.write(word)
                    out_f.write('\n')
                    mmap_file[count] = list(map(float, pieces[1:]))
                    count += 1
    
    print(len(seen_words))


if __name__ == '__main__':
    convert_word_vec()
