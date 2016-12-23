#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import customized_window_iterator
import numpy as np
import util

class WindowIteratorForUnitTest(customized_window_iterator.WindowIterator):
    def index_sequence_generator(self, path=""):
        for i in range(100):
            yield i

    def __init__(self, window, batch_size, repeat=True):
        super(WindowIteratorForUnitTest, self).__init__(window, batch_size, repeat)


class TestWindowIterator(unittest.TestCase):

    def test_init(self):
        window = 2 
        batch_size = 5 
        witerator = WindowIteratorForUnitTest(window, batch_size)
        
        self.assertTrue(witerator.window == window)
        self.assertTrue(witerator.batch_size, batch_size)
        self.assertTrue(witerator.total_size == util.count_total_size(witerator.index_sequence_generator(), ""))
        self.assertTrue(witerator.batch_index == 0)
        upper_count = 2 
        try:
            for center, context in witerator:
                for v, cs in zip(center, context):
                    indices = [index for index in range(-window, window+1) if index != 0]
                    for c, i in zip(cs, indices):
                        self.assertTrue(c == v + i)
                if upper_count == witerator.epoch_detail:
                    break
        except Exception as e:
            print(e.message)
     

if __name__ == "__main__":
    unittest.main()

