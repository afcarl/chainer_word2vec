#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class WordSearcher(object):

    def __init__(self, n_results):
        self.n_results = n_results

    def load_model(self, model_path):
        with open(model_path, 'r') as f:
            ss = f.readline().split()
            n_vocab, n_units = int(ss[0]), int(ss[1])
            self.word2index = {}
            self.index2word = {}
            w = np.empty((n_vocab, n_units), dtype=np.float32)
            for i, line in enumerate(f):
                ss = line.split()
                assert len(ss) == n_units + 1
                word = ss[0]
                self.word2index[word] = i
                self.index2word[i] = word
                w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

        s = np.sqrt((w * w).sum(1))
        w /= s.reshape((s.shape[0], 1))  # normalize
        self.w = w

    # test ok
    def search(self, query):
        similar_words = []
        if query not in self.word2index:
            print('"{0}" is not found'.format(query))
            return similar_words

        v = self.w[self.word2index[query]]
        similarity = self.w.dot(v)
        print('query: {}'.format(query))
        count = 0
        for i in (-similarity).argsort():
            if np.isnan(similarity[i]):
                continue
            if self.index2word[i] == query:
                continue
            # print('{0}: {1}'.format(self.index2word[i], similarity[i]))
            similar_words.append((self.index2word[i], similarity[i]))
            count += 1
            if count == self.n_results:
                break
        return similar_words


if __name__ == '__main__':
    pass
