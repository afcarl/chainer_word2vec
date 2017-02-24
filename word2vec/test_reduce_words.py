#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import cPickle
import os


WORD2INDEX_PATH = "/Users/kumada/Data/enwiki/20160920_word2index_min_count_32.pkl"
INDEX2WORD_PATH = "/Users/kumada/Data/enwiki/20160920_index2word_min_count_32.pkl"


class reduce_words_Test(unittest.TestCase):

    def test_check(self):
        assert os.path.exists(WORD2INDEX_PATH), ""
        assert os.path.exists(INDEX2WORD_PATH), ""

        w2i = cPickle.load(open(WORD2INDEX_PATH))
        i2w = cPickle.load(open(INDEX2WORD_PATH))
        print(len(w2i))
        i = w2i["<unk>"]
        w = i2w[i]
        self.assertTrue(w == "<unk>")


if __name__ == "__main__":
    unittest.main()
