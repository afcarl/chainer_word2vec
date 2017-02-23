#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import argparse
import util


def word_generator(path):
    for line in open(path):
        items = line.strip().split()
        yield items[1], items[2]  # word, image_num


if __name__ == "__main__":
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_path", help="input: set a path to a space-separted text file")
        parser.add_argument("--word2index_path", help="input: set a path a word2index file(.pkl)")
        parser.add_argument("--output_path", help="output: set an output path")

        # parse arguments
        args = parser.parse_args()
        file_path = args.file_path
        word2index_path = args.word2index_path
        output_path = args.output_path

        # check paths
        util.check_input_path(file_path)
        util.check_input_path(word2index_path)
        util.check_output_path(output_path)

        print("> now loading word2index...")
        word2index = cPickle.load(open(word2index_path))
        print("> loading done!")

        with open(output_path, "w") as output:
            for word, num in word_generator(file_path):
                if word in word2index:
                    output.write("{} {}\n".format(word, num))

    except IOError, e:
        print(e)
