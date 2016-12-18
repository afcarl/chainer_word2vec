#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle


def load_data(filename):
    unique_words = set()
    with open(FILE_PATH) as fin:
        for line in fin:
            items = line.strip().split()
            register_words(unique_words, items)

    words = list(unique_words)
    word2index = {}
    index2word = {}
    for i, word in enumerate(words):
        word2index[word] = i
        index2word[i] = word

    return word2index, index2word


def register_words(uwords, words):
    for word in words:
        uwords.add(word)


FILE_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati.txt"
WORD2INDEX_PATH = "/home/ubuntu/data/word2vec/word2index.pkl"
INDEX2WORD_PATH = "/home/ubuntu/data/word2vec/index2word.pkl"

if __name__ == "__main__":

    word2index, index2word = load_data(FILE_PATH)
    cPickle.dump(word2index, open(WORD2INDEX_PATH, "wb"))
    cPickle.dump(index2word, open(INDEX2WORD_PATH, "wb"))
