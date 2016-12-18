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
import window_iterator

import os
import cPickle
import index_sequence_maker

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


def count_total_size(input_path, output_path):
    if os.path.exists(output_path):
        return cPickle.load(open(output_path))
    else:
        c = 0
        for index in index_sequence_maker.index_generator(input_path):
            c += 1
        cPickle.dump(c, open(output_path, "wb"))
        return c


def find_train_max(input_path, max_count, output_path):
    if os.path.exists(output_path):
        return cPickle.load(open(output_path))
    else:
        max_index = -1
        for i, index in enumerate(index_sequence_maker.index_generator(input_path), start=1):
            if index > max_index:
                max_index = index
            if i == max_count:
                break
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


COUNTS_PATH = "/home/ubuntu/data/word2vec/counts.pkl"
TRAIN_MAX_PATH = "/home/ubuntu/data/word2vec/train_max.pkl"   
TOTAL_SIZE_PATH = "/home/ubuntu/data/word2vec/total_size.pkl"   


if __name__ == "__main__":

    parser = make_argparser()
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Training model: {}'.format(args.model))
    print('Output type: {}'.format(args.out_type))
    print('input: {}'.format(args.input))
    print('index2word: {}'.format(args.index2word))
    print('word2index: {}'.format(args.word2index))
    print('')


    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    total_size = count_total_size(args.input, TOTAL_SIZE_PATH)
    train_size = int(total_size * 0.7)
    val_size = total_size - train_size
#
#    train, val, _ = chainer.datasets.get_ptb_words()
#    counts = collections.Counter(train)
#    counts.update(collections.Counter(val))
     
    counts = make_counter(args.input, COUNTS_PATH)
    train_max = find_train_max(args.input, train_size, TRAIN_MAX_PATH)
    n_vocab = train_max + 1
#    n_vocab = max(train) + 1

#    if args.test:
#        train = train[:100]
#        val = val[:100]
#
#    vocab = chainer.datasets.get_ptb_words_vocabulary()
    vocab = cPickle.load(open(args.word2index)) 
    index2word = cPickle.load(open(args.index2word))
    index = vocab["東京"]
    print(index2word[index])
#
#    print('n_vocab: %d' % n_vocab)
#    print('data length: %d' % len(train))

    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % train_size)
    

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
        n_vocab = 900000
        model = continuous_bow.ContinuousBoW(n_vocab, args.unit, loss_func)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()
#
#
#    optimizer = O.Adam()
#    optimizer.setup(model)
#
#    train_iter = window_iterator.WindowIterator(train, args.window, args.batchsize)
#    val_iter = window_iterator.WindowIterator(val, args.window, args.batchsize, repeat=False)
#    updater = training.StandardUpdater(
#        train_iter, optimizer, converter=convert, device=args.gpu)
#    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
#
#    trainer.extend(extensions.Evaluator(
#        val_iter, model, converter=convert, device=args.gpu))
#    trainer.extend(extensions.LogReport())
#    trainer.extend(extensions.PrintReport(
#        ['epoch', 'main/loss', 'validation/main/loss']))
#    trainer.extend(extensions.ProgressBar())
#    trainer.run()
#
#    with open('word2vec.model', 'w') as f:
#        f.write('%d %d\n' % (len(index2word), args.unit))
#        w = cuda.to_cpu(model.embed.W.data)
#        for i, wi in enumerate(w):
#            v = ' '.join(map(str, wi))
#            f.write('%s %s\n' % (index2word[i], v))