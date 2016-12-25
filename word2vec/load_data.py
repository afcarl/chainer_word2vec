#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import collections


def create_histogram(filename):
    histogram = collections.defaultdict(int)
    with open(FILE_PATH) as fin:
        for line in fin:
            words = line.strip().split()
            register_words(histogram, words)
    return histogram


def register_words(histogram, words):
    for word in words:
        histogram[word] += 1


UNKNOWN_WORD = "<unk>"
#MIN_COUNT = 9  # small: We have to make the number of words be less than 661000 to avoid the cuda memory error.
#MIN_COUNT = 23  # original: We have to make the number of words be less than 661000 to avoid the cuda memory error.
MIN_COUNT = 5

# If frequency of a word is less than "min_count,"  the word is removed.
def reduce_words(histogram, min_count):
    for (key, value) in histogram.items():
        if value < min_count:
            del histogram[key]
    # add <unk>, which means "unknown word."
    histogram[UNKNOWN_WORD] += 1


def make_maps(histogram):
    word2index = {}
    index2word = {}
    for i, word in enumerate(histogram.keys()):
        word2index[word] = i
        index2word[i] = word
    return word2index, index2word


FILE_PATH = "/home/ubuntu/data/word2vec/small/jawiki-wakati.txt"
WORD2INDEX_PATH = "/home/ubuntu/data/word2vec/small/word2index.pkl"
INDEX2WORD_PATH = "/home/ubuntu/data/word2vec/small/index2word.pkl"


def save(file_path, word2index_path, index2word_path):
    histogram = create_histogram(FILE_PATH)
    print("The total number of words is {}.".format(len(histogram)))
    reduce_words(histogram, min_count=MIN_COUNT)
    assert 1 == histogram[UNKNOWN_WORD], ""
    print("The number of words is {} after the reduction.".format(len(histogram)))
    (word2index, index2word) = make_maps(histogram)
    cPickle.dump(word2index, open(WORD2INDEX_PATH, "wb"))
    cPickle.dump(index2word, open(INDEX2WORD_PATH, "wb"))


def check(word2index_path, index2word_path):
    word2index = cPickle.load(open(word2index_path))
    index2word = cPickle.load(open(index2word_path))
    print(len(word2index))
    print(len(index2word))
    index = word2index["<unk>"]
    print(index, index2word[index])


if __name__ == "__main__":
    save(FILE_PATH, WORD2INDEX_PATH, INDEX2WORD_PATH)
    # check(WORD2INDEX_PATH, INDEX2WORD_PATH)
