#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import argparse
import util


def word_generator(path):
    for line in open(path):
        line = line.strip()
        yield line


if __name__ == "__main__":
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_path", help="input: set a path to a space-separted text file")
        parser.add_argument("--word2index_path", help="output: set a path a word2index file(.pkl)")

        # parse arguments
        args = parser.parse_args()
        file_path = args.file_path
        word2index_path = args.word2index_path

        # check paths
        util.check_input_path(file_path)
        util.check_input_path(word2index_path)

        print("> now loading word2index...")
        word2index = cPickle.load(open(word2index_path))
        print("> loading done!")

        for word in word_generator(file_path):
            if word in word2index:
                print("{} -> {}".format(word, word2index[word]))
            else:
                print("ERROR: {} is not found".format(word))

    except IOError, e:
        print(e)
