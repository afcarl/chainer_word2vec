#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import reporter

class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__(
            embed=L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units)),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        e = self.embed(context)
        shape = e.data.shape
        x = F.broadcast_to(x[:, None], (shape[0], shape[1]))
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        x = F.reshape(x, (shape[0] * shape[1],))
        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


