#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import index_sequence_maker
import cPickle
import load_data

WORD2INDEX_FILE_PATH = "/Users/kumada/Data/enwiki/20160920_word2index.pkl"  # input
INDEX2WORD_FILE_PATH = "/Users/kumada/Data/enwiki/20160920_index2word.pkl"  # input
WIKI_FILE_PATH = "/Users/kumada/Data/enwiki/unittest/20160920.wiki"  # input
INDEX_SEQUENCE_FILE_PATH = "/Users/kumada/Data/enwiki/unittest/20160920_index_sequence.txt"  # output


class index_sequende_Test(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(index_sequende_Test, self).__init__(*args, **kwargs)
        self.word2index = cPickle.load(open(WORD2INDEX_FILE_PATH))
        self.index2word = cPickle.load(open(INDEX2WORD_FILE_PATH))

    def test_get_index(self):
        token = "kumada"
        index = index_sequence_maker.get_index(self.word2index, token)
        # self.assertTrue(load_data.UNKNOWN_WORD == self.index2word[index])
        self.assertTrue(token == self.index2word[index])

        token = "economical"
        index = index_sequence_maker.get_index(self.word2index, token)
        self.assertTrue(token == self.index2word[index])

        token = "seiya"
        index = index_sequence_maker.get_index(self.word2index, token)
        # self.assertTrue(load_data.UNKNOWN_WORD == self.index2word[index])
        self.assertTrue(token == self.index2word[index])

        token = "す"
        index = index_sequence_maker.get_index(self.word2index, token)
        self.assertTrue(load_data.UNKNOWN_WORD == self.index2word[index])

    def test_save(self):
        index_sequence_maker.save(WIKI_FILE_PATH, WORD2INDEX_FILE_PATH, INDEX_SEQUENCE_FILE_PATH)
        with open(INDEX_SEQUENCE_FILE_PATH) as fin:
            lines = fin.readlines()
            a = lines[0].split()
            b = lines[1].split()
            c = lines[2].split()
            self.assertTrue(int(a[28]) == 5414296)
            self.assertTrue(int(b[39]) == 4324268)
            self.assertTrue(int(c[8]) == 6135003)


if __name__ == "__main__":
    unittest.main()
