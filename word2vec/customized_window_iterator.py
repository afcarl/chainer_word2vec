#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer


class WindowIterator(chainer.dataset.Iterator):

#    def __init__(self, dataset, window, batch_size, repeat=True):
#        self.dataset = np.array(dataset, np.int32)
#        self.window = window
#        self.batch_size = batch_size
#        self._repeat = repeat
#
#        self.order = np.random.permutation(
#            len(dataset) - window * 2).astype(np.int32)
#            # 17 - 4 = 13 [0,13)
#        self.order += window # [2,15) 2,3,...,13,14
#        self.current_position = 0
#        self.epoch = 0
#        self.is_new_epoch = False
#
#
#    def __next__(self):
#        if not self._repeat and self.epoch > 0:
#            raise StopIteration
#
#        i = self.current_position # 0
#        i_end = i + self.batch_size # 5
#        position = self.order[i: i_end] # order[0:5]
#        w = np.random.randint(self.window - 1) + 1 # 1
#        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
#        # [-2,0) + [1,3) = [-2,-1,1,2] 
#        pos = position[:, None] + offset[None, :]
#        context = self.dataset.take(pos)
#        center = self.dataset.take(position)
#
#        if i_end >= len(self.order):
#            np.random.shuffle(self.order)
#            self.epoch += 1
#            self.is_new_epoch = True
#            self.current_position = 0
#        else:
#            self.is_new_epoch = False
#            self.current_position = i_end
#
#        return center, context

    def count_total_size(self, generator):
        return sum([1 for _ in generator])

    def __init__(self, window, batch_size, repeat=True):
        self.generator = self.index_sequence_generator()
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        total_size = self.count_total_size(self.index_sequence_generator()) # 17
        #self.upper_size = total_size - window * 2 # 17 - 4 = 13 
        #self.allocated_size = 2 * self.window + self.batch_size # 4 + 5 = 9
        self.batch_index = 0
        self.sequence_head = np.array([next(self.generator) for _ in range(2 * self.windows)]) # [0,1,2,3]
        self.is_end_of_epoch = False
        w = np.random.randint(self.window - 1) + 1 # 2
        self.offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)]) # [-2,-1,1,2]

    def index_sequence_generator(self):
        for i in range(17):
            yield i

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        remained_size = total_size - (self.batch_size * self.batch_index + 2 * self.window) # 17-(10+4)=3
        allocated_size = min(remained_size, self.batch_size) # 3,5 -> 3 
        if allocated_size == remained_size:
            is_end_of_epoch = True

        sequence = np.array([next(self.generator) for _ in range(allocated_size)]) # [14,15,16]
        sequence = np.r_[self.sequence_head, sequence] # [10,11,12,13,14,15,16]
       
        tmp = len(sequence) - 2*self.window # 7-4=3  
        order = np.random.shuffle(sequences[0:tmp]) + self.window
        # [10,11,12] -> [12,13,14]
        position = order  
        pos = position[:, None] + self.offset[None, :]
        context = self.dataset.take(pos)
        center = self.dataset.take(position)

        if is_end_of_epoch:
            self.epoch += 1
            self.generator = self.index_sequence_generator()
            self.batch_index = 0
        else:
            self.sequence_head = sequence[self.batch_size:self.batch_size+2*self.window] # [10,11,12,13]
            self.batch_index += 1 # 2 

        return center, context



    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


