#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions
import continuous_bow
import skip_gram
import softmax_cross_entropy_loss
import customized_window_iterator

import os
import util 
import cPickle
import index_sequence_maker
import make_dataset 

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of units')
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=2, type=int, # 20
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                        default='skipgram',
                        help='model type ("skipgram", "cbow")')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                        default='hsm',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, "original": no approximation)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--input', '-i', type=str, help='input file path')
    parser.add_argument('--index2word', type=str, help='index2word file path')
    parser.add_argument('--word2index', type=str, help='word2index file path')
    parser.set_defaults(test=False)
    return parser


def convert(batch, device):
    center, context = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        context = cuda.to_gpu(context)
    return center, context


def load_dataset(input_path, train_size, output_path_0, output_path_1):
    if os.path.exists(output_path_0) and os.path.exists(output_path_1):
        total = make_dataset.load_dataset(output_path_0, output_path_1)
    else:
        total = make_dataset.save_dataset(input_path, output_path_0, output_path_1)
    train = total[:train_size]
    val = total[train_size:]
    return (train, val)


COUNTS_PATH = "/home/ubuntu/data/word2vec/small/counts.pkl"
TRAIN_MAX_PATH = "/home/ubuntu/data/word2vec/small/train_max.pkl"   
TOTAL_SIZE_PATH = "/home/ubuntu/data/word2vec/small/total_size.pkl"   
COUNTS_PATH = "/home/ubuntu/data/word2vec/small/counts.pkl"   


def make_dummy_counts(size):
    return collections.Counter({i: i for i in range(size)})


if __name__ == "__main__":

    parser = make_argparser()
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('negative size: {}'.format(args.negative_size))
    print('input: {}'.format(args.input))
    print('index2word: {}'.format(args.index2word))
    print('word2index: {}'.format(args.word2index))
    print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    
    total_size = util.count_total_size(args.input, TOTAL_SIZE_PATH)
    print("total_size: {}".format(total_size))

    print("...load counts")
    counts = util.make_counter(args.input, COUNTS_PATH)
    train_max = util.find_train_max(args.input, TRAIN_MAX_PATH)
    n_vocab = train_max + 1

    print("...load vocab and index2word")
    vocab = cPickle.load(open(args.word2index)) 
    index2word = cPickle.load(open(args.index2word))
    print("n_vocab: {}".format(n_vocab))
    print("counts: {}".format(len(counts)))

    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        tree = HSM.create_huffman_tree(counts)
        loss_func = HSM(args.unit, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [counts[w] for w in range(len(counts))]
        loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = softmax_cross_entropy_loss.SoftmaxCrossEntropyLoss(args.unit, n_vocab)
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))

    if args.model == 'skipgram':
        model = skip_gram.SkipGram(n_vocab, args.unit, loss_func)
    elif args.model == 'cbow':
        model = continuous_bow.ContinuousBoW(n_vocab, args.unit, loss_func)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        # When selecting GPU, the error of "out of memory" occurs.
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = customized_window_iterator.WindowIterator(
            args.window, 
            args.batchsize,
            index_sequence_file_path=args.input,
            total_size_path=TOTAL_SIZE_PATH,
            repeat=True)
            
    updater = training.StandardUpdater(
        train_iter, 
        optimizer, 
        converter=convert, 
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    with open('word2vec.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))
