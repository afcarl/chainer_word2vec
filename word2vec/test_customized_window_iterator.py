#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import customized_window_iterator
import numpy as np

class TestWindowIterator(unittest.TestCase):
    def index_generator(self, last_index):
        for i in range(last_index):
            yield i

#
#    def test_init(self):
#        last_value = 17
#        dataset = np.arange(last_value) 
#        window = 2 
#        batch_size = 5 
#        witerator = customized_window_iterator.WindowIterator(dataset, window, batch_size)
#        upper_count = 10
#        for center, context in witerator:
#            #center, context = next(witerator)
#            for v, cs in zip(center, context):
#                indices = [index for index in range(-(window/2), window/2+1) if index != 0]
#                for c, i in zip(cs, indices):
#                    self.assertTrue(c == v + i)
#            #print(witerator.epoch_detail)
#            if upper_count == witerator.epoch_detail:
#                break

    def test_init(self):
        window = 2 
        batch_size = 5 
        witerator = customized_window_iterator.WindowIterator(window, batch_size)
        self.assertTrue(witerator.window == window)
        self.assertTrue(witerator.batch_size, batch_size)
        total_size = witerator.count_total_size(witerator.index_sequence_generator())
        self.assertTrue(witerator.upper_size == total_size - 2 * window)
        self.assertTrue(witerator.allocated_size == allocated_size)
        c = 0
        try:
            for center, context in witerator:
                #print(center, context)
                c += 1 
    #            #center, context = next(witerator)
    #            for v, cs in zip(center, context):
    #                indices = [index for index in range(-(window/2), window/2+1) if index != 0]
    #                for c, i in zip(cs, indices):
    #                    self.assertTrue(c == v + i)
    #            #print(witerator.epoch_detail)
    #            if upper_count == witerator.epoch_detail:
    #                break
    #            if c == 4:
    #                break
        except Error as e:
            print(e.message)
     

if __name__ == "__main__":
    unittest.main()

