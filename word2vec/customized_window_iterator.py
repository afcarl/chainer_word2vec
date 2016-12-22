#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer


class WindowIterator(chainer.dataset.Iterator):

    def count_total_size(self, generator):
        return sum([1 for _ in generator])

    def __init__(self, window, batch_size, repeat=True):
        self.generator = self.index_sequence_generator()
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        self.total_size = self.count_total_size(self.index_sequence_generator())
        self.double_window = 2 * self.window
        self.batch_index = 0
        self.sequence_head = self.make_sequence_head(self.generator, self.double_window)
        self.is_end_of_epoch = False
        w = self.window # np.random.randint(self.window - 1) + 1 # 2
        self.offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        self.current_position = 0

    def make_sequence_head(self, generator, size):
        return np.array([next(generator) for _ in range(size)])
    
    def index_sequence_generator(self):
        for i in range(100):
            yield i

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
            self.generator = self.index_sequence_generator()
            self.sequence_head = self.make_sequence_head(self.generator, self.double_window) 
            self.batch_index = 0
            self.is_end_of_epoch = False
        else:
            self.sequence_head = sequence[self.batch_size:self.batch_size + self.double_window]
            self.batch_index += 1

        return center, context

    @property
    def epoch_detail(self):
        fraction = float(self.current_position) / self.total_size
        return self.epoch + (fraction if fraction < 1 else 0)

    def serialize(self, serializer):
        pass
