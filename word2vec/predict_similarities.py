#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np


def check_path(path):
    if not os.path.exists(path):
        raise IOError('invalid path')


def load_model(model_path):

    with open(args.model_path) as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        w2i = {}
        i2w = {}
        ww = np.empty((n_vocab, n_units), dtype=np.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            w2i[word] = i
            i2w[i] = word
            ww[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

    return w2i, i2w, ww


def show_similar_words(sim, i2w, count):
    count = 0
    for i in (-sim).argsort():
        if np.isnan(sim[i]):
            continue
        if i2w[i] == q:
            continue
        print('{0}: {1}'.format(i2w[i], sim[i]))
        count += 1
        if count == args.count:
            break


if __name__ == '__main__':
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--label_path', help='input: a path to a label file')
        parser.add_argument('--model_path', help='input: a path to a model file')
        parser.add_argument('--count', type=int, help='input: number of search result to show')

        # parse arguments
        args = parser.parse_args()
        check_path(args.label_path)
        check_path(args.model_path)

        word2index, index2word, w = load_model(args.model_path)

        s = np.sqrt((w * w).sum(1))
        w /= s.reshape((s.shape[0], 1))  # normalize

        for line in open(args.label_path):
            tokens = line.strip().split()
            q = tokens[0]
            if q not in word2index:
                print('"{0}" is not found'.format(q))
                continue
            v = w[word2index[q]]
            similarity = w.dot(v)
            print('> query: {}'.format(q))
            show_similar_words(similarity, index2word, args.count)

    except IOError, e:
        print(e)
    except EOFError, e:
        print(e)
