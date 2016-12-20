#!/usr/bin/env python
# -*- coding: utf-8 -*-

import util
import index_sequence_maker
import numpy as np
import cPickle 

INDEX_SEQUENCE_FILE_PATH = "/home/ubuntu/data/word2vec/small/jawiki-wakati-index-sequence.txt"
TOTAL_DATASET_PATH = "/home/ubuntu/data/word2vec/small/dataset.pkl"   


def save_dataset(input_path, output_path):
    dataset = np.array([index for index in index_sequence_maker.index_generator(input_path)])
    cPickle.dump(dataset, open(output_path, "wb"))
    return dataset

def load_dataset(input_path_0, input_path_1):
    dataset_0 = cPickle.load(open(input_path_0))
    dataset_1 = cPickle.load(open(input_path_1))
    return np.r_[dataset_0, dataset_1]


if __name__ == "__main__":
    save_dataset(INDEX_SEQUENCE_FILE_PATH, TOTAL_DATASET_PATH)
    # dataset = load_dataset(TOTAL_DATASET_PATH_0, TOTAL_DATASET_PATH_1)

