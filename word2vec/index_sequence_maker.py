#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import sys

WORD2INDEX_FILE_PATH = "/home/ubuntu/data/word2vec/word2index.pkl"
JAWIKI_WAKATI_FILE_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati.txt"
INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati-index-sequence.txt"


def save(index_sequence_file_path):   
    fout = open(index_sequence_file_path, "w")
    c = 0
    word2index = cPickle.load(open(WORD2INDEX_FILE_PATH))
    for line in open(JAWIKI_WAKATI_FILE_PATH):
        tokens = line.strip().split()
        indices = [str(word2index[token]) for token in tokens]
        seq = " ".join(indices)
        fout.write("{}\n".format(seq)) 


def index_generator(index_sequence_file_path):
    for line in open(index_sequence_file_path):
        tokens = line.strip().split()
        for token in tokens:
            yield int(token)

if __name__ == "__main__":
    # save(INDEX_SEQUENCE_FILE_PATH)
    for (i, index) in enumerate(index_generator(INDEX_SEQUENCE_FILE_PATH)):
        print(index)
        if i == 10:
            break
