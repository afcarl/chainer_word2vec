#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import index_sequence_maker
import cPickle
import load_data

WORD2INDEX_FILE_PATH = "/home/ubuntu/data/word2vec/small/word2index.pkl" # input
INDEX2WORD_FILE_PATH = "/home/ubuntu/data/word2vec/small/index2word.pkl" # input
JAWIKI_WAKATI_FILE_PATH = "/home/ubuntu/data/word2vec/unittest/jawiki-wakati.txt" # input
INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/unittest/jawiki-wakati-index-sequence.txt" # output


class index_sequende_Test(unittest.TestCase):
    def test_get_index(self):
        word2index = cPickle.load(open(WORD2INDEX_FILE_PATH))
        index2word = cPickle.load(open(INDEX2WORD_FILE_PATH))
        token = "mimi"
        index = index_sequence_maker.get_index(word2index, token)
        self.assertTrue(load_data.UNKNOWN_WORD == index2word[index])

        token = "東京"
        index = index_sequence_maker.get_index(word2index, token)
        self.assertTrue(token == index2word[index])

    def test_save(self):
        word2index = cPickle.load(open(WORD2INDEX_FILE_PATH))
        index_sequence_maker.save(JAWIKI_WAKATI_FILE_PATH, WORD2INDEX_FILE_PATH, INDEX_SEQUENCE_FILE_PATH)
        with open(INDEX_SEQUENCE_FILE_PATH) as fin:
            lines = fin.readlines()
            a = lines[5].split()
            b = lines[6].split()
            c = lines[9].split()
            self.assertTrue(int(a[28]) == word2index[load_data.UNKNOWN_WORD])
            self.assertTrue(int(a[39]) == word2index[load_data.UNKNOWN_WORD])
            self.assertTrue(int(b[21]) == word2index[load_data.UNKNOWN_WORD])
            self.assertTrue(int(c[8]) == word2index[load_data.UNKNOWN_WORD])

            

if __name__ == "__main__":
    unittest.main()


