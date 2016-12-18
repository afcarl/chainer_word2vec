#!/usr/bin/env python
# -*- coding: utf-8 -*-


import six
import chainer

if __name__ == "__main__":
    train, val, test = chainer.datasets.get_ptb_words()
    word2index = chainer.datasets.get_ptb_words_vocabulary()
    index2word = {wid: word for word, wid in six.iteritems(word2index)}
    print(index2word[0])
    print(word2index["aer"])

