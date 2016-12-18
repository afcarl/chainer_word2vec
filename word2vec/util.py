#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle

def count_total_size(input_path, output_path):
    if os.path.exists(output_path):
        return cPickle.load(open(output_path))
    else:
        c = 0
        for index in index_sequence_maker.index_generator(input_path):
            c += 1
        cPickle.dump(c, open(output_path, "wb"))
        return c


if __name__ == "__main__":
    pass

