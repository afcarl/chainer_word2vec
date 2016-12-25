#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle
import index_sequence_maker
import collections

def count_total_size(input_path, out_path):
    if os.path.exists(out_path):
        return cPickle.load(open(out_path))
    else:
        generator = index_sequence_maker.index_generator(input_path)
        total_size = sum([1 for _ in generator])
        if out_path != "":
            cPickle.dump(total_size, open(out_path, "wb"))
        return total_size 


def find_train_max(input_path, output_path):
    if os.path.exists(output_path):
        return cPickle.load(open(output_path))
    else:
        max_index = -1
        for index in index_sequence_maker.index_generator(input_path):
            if index > max_index:
                max_index = index
        cPickle.dump(max_index, open(output_path, "wb"))
        return max_index 


def make_counter(input_path, output_path):
    if os.path.exists(output_path):
        return cPickle.load(open(output_path))
    else:
        dic = collections.defaultdict(int) 
        for index in index_sequence_maker.index_generator(input_path):
             dic[index] += 1 
        ret = collections.Counter(dic) 
        cPickle.dump(ret, open(output_path, "wb"))
        return ret




if __name__ == "__main__":
    pass

