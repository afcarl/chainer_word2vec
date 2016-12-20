#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import sys

WORD2INDEX_FILE_PATH = "/home/ubuntu/data/word2vec/small/word2index.pkl" # input
JAWIKI_WAKATI_FILE_PATH = "/home/ubuntu/data/word2vec/small/jawiki-wakati.txt" # input
INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/small/jawiki-wakati-index-sequence.txt" # output


def save(jawiki_wakati_file_path, word2index_file_path, index_sequence_file_path):   
    fout = open(index_sequence_file_path, "w")
    c = 0
    word2index = cPickle.load(open(word2index_file_path))
    for line in open(jawiki_wakati_file_path):
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
    save(JAWIKI_WAKATI_FILE_PATH, WORD2INDEX_FILE_PATH, INDEX_SEQUENCE_FILE_PATH)
