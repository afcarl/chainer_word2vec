#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import load_data
import argparse


# WORD2INDEX_FILE_PATH = "/home/ubuntu/data/word2vec/small/word2index.pkl"  # input
# JAWIKI_WAKATI_FILE_PATH = "/home/ubuntu/data/word2vec/original/jawiki-wakati.txt"  # input
# INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/original_with_small_vocabulary/jawiki-wakati-index-sequence.txt"  # output


# test ok
def get_index(word2index, token):
    return word2index[token] if token in word2index else word2index[load_data.UNKNOWN_WORD]


# test ok
def save(jawiki_wakati_file_path, word2index_file_path, index_sequence_file_path):
    fout = open(index_sequence_file_path, "w")
    word2index = cPickle.load(open(word2index_file_path))
    for line in open(jawiki_wakati_file_path):
        tokens = line.strip().split()
        indices = [str(get_index(word2index, token)) for token in tokens]
        seq = " ".join(indices)
        fout.write("{}\n".format(seq))


def index_generator(index_sequence_file_path):
    for line in open(index_sequence_file_path):
        tokens = line.strip().split()
        for token in tokens:
            yield int(token)


if __name__ == "__main__":
    # set command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2index_file_path", help="input: set a path to a word2index file")
    parser.add_argument("--wiki_file_path", help="input: set a path to a wiki file")
    parser.add_argument("--index_sequence_file_path", help="output: set a path to a index sequence file")

    # parse arguments
    args = parser.parse_args()
    word2index_file_path = args.word2index_file_path
    wiki_file_path = args.wiki_file_path
    index_sequence_file_path = args.index_sequence_file_path

    save(wiki_file_path, word2index_file_path, index_sequence_file_path)
