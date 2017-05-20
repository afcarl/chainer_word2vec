#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import cPickle


def check_path(path):
    if not os.path.exists(path):
        raise IOError('invalid path')


if __name__ == '__main__':
    try:
        # set command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--label_path', help='input: a path to a label file')
        parser.add_argument('--word2index_path', help='input: a path to a word2index file')

        # parse arguments
        args = parser.parse_args()
        check_path(args.label_path)
        check_path(args.word2index_path)

        word2index = cPickle.load(open(args.word2index_path))
        for line in open(args.label_path):
            tokens = line.strip().split()
            word = tokens[0]
            if word not in word2index:
                print('invaid word: {}'.format(word))

    except IOError, e:
        print(e)
