#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F

class SoftmaxCrossEntropyLoss(chainer.Chain):

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            out=L.Linear(n_in, n_out, initialW=0),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


