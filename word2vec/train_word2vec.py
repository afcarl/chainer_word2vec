#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
Use ../ptb/download.py to download 'ptb.train.txt'.
"""
import argparse
import collections

import chainer
from chainer import cuda
import chainer.links as L
import chainer.optimizers as Op
from chainer import training
from chainer.training import extensions
import continuous_bow
import skip_gram
import softmax_cross_entropy_loss
import customized_window_iterator

import os
import util
import cPickle
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
    parser.add_argument('--epoch', '-e', default=2, type=int,  # 20
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
    parser.add_argument('--input_path', '-i', type=str, help='input file path')
    parser.add_argument('--index2word_path', type=str, help='index2word file path')
    parser.add_argument('--word2index_path', type=str, help='word2index file path')
    parser.add_argument('--counts_path', type=str, help='counts file path')
    parser.add_argument('--train_max_path', type=str, help='train max file path')
    parser.add_argument('--total_size_path', type=str, help='total size file path')
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
    print('input_path: {}'.format(args.input_path))
    print('index2word_path: {}'.format(args.index2word_path))
    print('word2index_path: {}'.format(args.word2index_path))
    print('word2index_path: {}'.format(args.word2index_path))
    print('counts_path: {}'.format(args.counts_path))
    print('train_max_path: {}'.format(args.train_max_path))
    print('total_size_path: {}'.format(args.total_size_path))
    print('')

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    total_size = util.count_total_size(args.input_path, args.total_size_path)
    print("total_size: {}".format(total_size))

    print("...load counts")
    counts = util.make_counter(args.input_path, args.counts_path)
    train_max = util.find_train_max(args.input_path, args.train_max_path)
    n_vocab = train_max + 1

    print("...load vocab and index2word")
    vocab = cPickle.load(open(args.word2index_path))
    index2word = cPickle.load(open(args.index2word_path))
    print("n_vocab: {}".format(n_vocab))  # 8651374
    print("counts: {}".format(len(counts)))  # 8651373

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

    optimizer = Op.Adam()
    optimizer.setup(model)

    train_iter = customized_window_iterator.WindowIterator(
        args.window,
        args.batchsize,
        index_sequence_file_path=args.input_path,
        total_size_path=args.total_size_path,
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
