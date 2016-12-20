#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import customized_window_iterator
import numpy as np

class TestWindowIterator(unittest.TestCase):
    def index_generator(self, last_index):
        for i in range(last_index):
            yield i

    def test_init(self):
        last_value = 17
        dataset = np.arange(last_value) 
        window = 2 
        batch_size = 5 
        witerator = customized_window_iterator.WindowIterator(dataset, window, batch_size)
        center, context = next(witerator)
        for v, cs in zip(center, context):
            indices = [index for index in range(-(window/2), window/2+1) if index != 0]
            for c, i in zip(cs, indices):
                self.assertTrue(c == v + i)
        
#        position = np.arange(3)
#        print(position.shape)
#        offset = np.array([-2,-1,1,2])
#        pos = position[:,None] + offset[None,:]
#        print(pos)



if __name__ == "__main__":
    unittest.main()

