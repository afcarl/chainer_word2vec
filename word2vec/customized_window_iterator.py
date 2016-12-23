#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import index_sequence_maker
import os
import cPickle
import util

class WindowIterator(chainer.dataset.Iterator):
    
    def __init__(
            self, 
            window, 
            batch_size, 
            index_sequence_file_path="", 
            total_size_path="",
            repeat=True):
        self.index_sequence_file_path = index_sequence_file_path
        self.generator = self.index_sequence_generator(self.index_sequence_file_path)
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        self.total_size = util.count_total_size(
                self.index_sequence_generator(self.index_sequence_file_path),
                total_size_path)
        self.double_window = 2 * self.window
        self.batch_index = 0
        self.sequence_head = self.make_sequence_head(self.generator, self.double_window)
        self.is_end_of_epoch = False
        w = self.window  # np.random.randint(self.window - 1) + 1 # 2
        self.offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        self.current_position = 0

    def make_sequence_head(self, generator, size):
        return np.array([next(generator) for _ in range(size)]).astype(np.int32)

    def index_sequence_generator(self, path):
        return index_sequence_maker.index_generator(self.index_sequence_file_path)

    # test ok
    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        remained_size = self.total_size - (self.batch_size * self.batch_index + self.double_window)
        allocated_size = min(remained_size, self.batch_size)
        self.current_position = allocated_size + self.batch_size * self.batch_index + self.double_window

        if allocated_size == remained_size:
            self.is_end_of_epoch = True

        sequence = self.make_sequence_head(self.generator, allocated_size)
        sequence = np.r_[self.sequence_head, sequence]
        tmp = len(sequence) - self.double_window
        order = np.random.permutation(tmp) + self.window
        position = order
        pos = position[:, None] + self.offset[None, :]
        context = sequence.take(pos)
        center = sequence.take(position)

        if self.is_end_of_epoch:
            self.epoch += 1
            self.generator = self.index_sequence_generator(self.index_sequence_file_path)
            self.sequence_head = self.make_sequence_head(self.generator, self.double_window)
            self.batch_index = 0
            self.is_end_of_epoch = False
        else:
            self.sequence_head = sequence[self.batch_size:self.batch_size + self.double_window]
            self.batch_index += 1

        print("epoch: {}".format(self.epoch_detail))
        return center, context

    @property
    def epoch_detail(self):
        fraction = float(self.current_position) / self.total_size
        return self.epoch + (fraction if fraction < 1 else 0)

    def serialize(self, serializer):
        pass

if __name__ == "__main__":
    import progressbar
    import time

    window = 8 
    batch_size = 10000 
    sequence_path = "/home/ubuntu/data/word2vec/small/jawiki-wakati-index-sequence.txt"
    total_size_path = "/home/ubuntu/data/word2vec/small/total_size.pkl"

    witerator = WindowIterator(
        window, 
        batch_size, 
        index_sequence_file_path=sequence_path, 
        total_size_path=total_size_path,
        repeat=True)
    
    print("witerator.total_size:{}".format(witerator.total_size) )
    upper_count = 1 
    count_scale = 100
    progressbar = progressbar.ProgressBar(maxvalue=int(count_scale * upper_count))
    indices = [index for index in range(-window, window+1) if index != 0]
    for center, context in witerator:
        progressbar.update(int(count_scale * witerator.epoch_detail))
        if upper_count == witerator.epoch_detail:
            break
