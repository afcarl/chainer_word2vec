#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import collections
import argparse
import util


UNKNOWN_WORD = "<unk>"


def create_histogram(file_path):
    histogram = collections.defaultdict(int)
    with open(file_path) as fin:
        for line in fin:
            words = line.strip().split()
            register_words(histogram, words)
    return histogram


def register_words(histogram, words):
    for word in words:
        histogram[word] += 1


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


def save(file_path, word2index_path, index2word_path, histogram_path, min_count):
    histogram = create_histogram(file_path)
    cPickle.dump(histogram, open(histogram_path, "wb"))
    print("> save histogram")

    print("The total number of words is {}.".format(len(histogram)))
    reduce_words(histogram, min_count=min_count)
    assert 1 == histogram[UNKNOWN_WORD], ""
    print("The number of words is {} after the reduction.".format(len(histogram)))
    (word2index, index2word) = make_maps(histogram)

    cPickle.dump(word2index, open(word2index_path, "wb"))
    print("> save word2index")

    cPickle.dump(index2word, open(index2word_path, "wb"))
    print("> save index2word")


def check(word2index_path, index2word_path):
    word2index = cPickle.load(open(word2index_path))
    index2word = cPickle.load(open(index2word_path))
    print(len(word2index))
    print(len(index2word))
    index = word2index["<unk>"]
    print(index, index2word[index])


if __name__ == "__main__":
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_path", help="input: set a path to a space-separted text file")
        parser.add_argument("--min_count", help="input: ignore a word whose frequency is less than it")
        parser.add_argument("--histogram_path", help="output: set a path a histogram(.pkl)")
        parser.add_argument("--word2index_path", help="output: set a path a word2index file(.pkl)")
        parser.add_argument("--index2word_path", help="output: set a path an index2word file(.pkl)")
        parser.add_argument("--check_mode", default=False, help="input: if True, data is checked")

        # parse arguments
        args = parser.parse_args()
        file_path = args.file_path
        word2index_path = args.word2index_path
        index2word_path = args.index2word_path
        histogram_path = args.histogram_path
        min_count = int(args.min_count)
        check_mode = args.check_mode

        util.check_input_path(file_path)
        if check_mode:
            util.check_input_path(word2index_path)
            util.check_input_path(index2word_path)
            util.check(word2index_path, index2word_path)
        else:
            util.check_output_path(word2index_path)
            util.check_output_path(index2word_path)
            util.check_output_path(histogram_path)
            save(file_path, word2index_path, index2word_path, histogram_path, min_count)

    except util.FileCanNotBeMadeError, e:
        print(e)
    except IOError, e:
        print(e)
