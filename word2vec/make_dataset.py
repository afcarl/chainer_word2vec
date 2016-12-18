#!/usr/bin/env python
# -*- coding: utf-8 -*-

import util
import index_sequence_maker
import numpy as np
import cPickle 

INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/jawiki-wakati-index-sequence.txt"
TOTAL_DATASET_PATH_0 = "/home/ubuntu/data/word2vec/total_dataset_0.pkl"   
TOTAL_DATASET_PATH_1 = "/home/ubuntu/data/word2vec/total_dataset_1.pkl"   


def save_dataset(input_path, output_path_0, output_path_1):
    dataset = np.array([index for index in index_sequence_maker.index_generator(input_path)])
    l = len(dataset)
    cPickle.dump(dataset[:l/2], open(output_path_0, "wb"))
    cPickle.dump(dataset[l/2:], open(output_path_1, "wb"))
    return dataset

def load_dataset(input_path_0, input_path_1):
    dataset_0 = cPickle.load(open(input_path_0))
    dataset_1 = cPickle.load(open(input_path_1))
    return np.r_[dataset_0, dataset_1]


if __name__ == "__main__":
    # save_dataset(INDEX_SEQUENCE_FILE_PATH, TOTAL_DATASET_PATH_0, TOTAL_DATASET_PATH_1)
    dataset = load_dataset(TOTAL_DATASET_PATH_0, TOTAL_DATASET_PATH_1)

