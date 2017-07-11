#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from word_searcher import WordSearcher


class TestWordSearcher(unittest.TestCase):

    def test_search(self):
        n_results = 5  # number of search result to show
        model_path = '/home/ubuntu/results/word2vec/word2vec.model'
        searcher = WordSearcher(n_results)
        searcher.load_model(model_path)
        similar_words = searcher.search('tokyo')
        answers = [('osaka', 0.88120759), ('nagoya', 0.82389867),
                   ('yokohama', 0.81960642), ('shibuya', 0.794792), ('shinjuku', 0.78090179)]

        for (similar_word, answer) in zip(similar_words, answers):
            self.assertTrue(similar_word[0] == answer[0])
            self.assertAlmostEqual(similar_word[1], answer[1], delta=1.0e-5)

        similar_words = searcher.search('microsoft')
        print(similar_words)


if __name__ == '__main__':
    unittest.main()
