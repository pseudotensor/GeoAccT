# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:57:34 2016

Experiments on Natural Gradient Descent with Geodesic Acceleration
@author: Jonathan C. McKinney
"""

import theano
import theano.tensor as T
import loadData
import nn
import numpy as np
import lincg
import argparse
import math

# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity='high'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments about geodesic acceleration")
    parser.add_argument('--lr', '-l', type=float, help="learning rate")
    parser.add_argument('--std', type=float, help="the standard deviation of weights initialization")
    parser.add_argument('--geo', action="store_true", help="turn on geodesic acceleration")
    parser.add_argument('--ratio', '-r', type=float, default=0.5,
                        help="the ratio of GeoAcc term to the natural gradient term")
    fake_arg = '--lr 1 --std 1'
    args = parser.parse_args(fake_arg.split())

    np.random.seed(6119843)

    train_set_x, train_set_y, valid_set_x, valid_set_y, \
    test_set_x, test_set_y, train_set_x_np, train_set_y_np = loadData.load('MNIST')

    data_size = 50000
    batch_size = 5000
    n_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    x = T.matrix('x')
    y = T.matrix('y')
    all_x = T.matrix('all_x')
    all_loss = T.fscalar('all_loss')
    size = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
    # size = [784, 500, 784]
    ae = nn.AutoEncoder(x, all_x, size, batchsize=batch_size, datasize=data_size, std=args.std)

    index = T.iscalar('index')
    cost = ae.get_cost(y)
    reconstruction_loss = ae.get_reconstruction_loss(y)
    givens = {
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_x[index * batch_size: (index + 1) * batch_size],
        all_x: train_set_x
    }
    '''
    train_error = theano.function([index], cost , givens=givens)
    valid_error = theano.function([index], cost, givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_x[index * batch_size: (index + 1) * batch_size]
    })
    '''
    test_error = theano.function([index], [reconstruction_loss, cost], givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_x[index * batch_size: (index + 1) * batch_size],
        all_x: train_set_x
    })

    initlambda = 45.0
    damp = theano.shared(np.asarray(initlambda, dtype=theano.config.floatX))
    decay = 0.95  # decay for CG initialization, following Martens, 2010.
    # gparams = [alpha * T.grad(cost, param) for param in ae.params]
    # grad, grad2 = ae.backprop(y)
    grad, grad2, all_ll = ae.backprop_all(args.lr)
    gparams = [x for x in grad]
    precond = [(x + damp) ** (3.0 / 4.0) for x in grad2]
    dottheta0 = [T.zeros_like(x) for x in ae.params]


    def Gvs(v):
        return [T.Lop(ae.output, p,
                      sum(T.Rop(ae.output, param, gparam) for param, gparam in zip(ae.params, v)) / (
                          ae.output * (1 - ae.output) * batch_size)) for p in ae.params]


    dottheta = lincg.linear_cg_precond(ae.metric_product, gparams, precond, *dottheta0, maxit=2)
    gradXdottheta =  sum(T.sum(x * y) for x, y in zip(grad, dottheta))
    dotthetaXGXdottheta = sum(T.sum(x * y) for x, y in zip(dottheta, ae.metric_product(dottheta)))
    # dottheta = lincg.linear_cg(ae.metric_product, gparams, *dottheta0, maxit=100, damp=damp)
    norm1 = T.sqrt(sum(T.sum(x ** 2) for x in dottheta))
    geoterm0 = [T.zeros_like(x) for x in ae.params]

    # geoterm is for squared loss
    # geoterm = lincg.linear_cg(ae.metric_product, ae.term1(dottheta), *geoterm0, maxit=20, damp=damp)
    # geoterm_xent is for cross-entropy loss
    geoterm_xent = lincg.linear_cg_precond(ae.metric_product,
                                           [x + y for x, y in zip(ae.term1(dottheta), ae.term2(dottheta))], precond,
                                           *geoterm0, maxit=2)
    gradXp = sum(T.sum(x * (t1 - args.ratio * t2)) for x, t1, t2 in zip(grad, dottheta, geoterm_xent))
    p = [t1 - args.ratio * t2 for t1, t2 in zip(dottheta, geoterm_xent)]
    pXGp = sum(T.sum(x * y) for x, y in zip(p, ae.metric_product(p)))
    #   geoterm_xent = lincg.linear_cg(ae.metric_product, [x + y for x, y in zip(ae.term1(dottheta), ae.term2(dottheta))],
    #                                          *geoterm0, maxit=100, damp=damp)

    norm2 = T.sqrt(sum(T.sum(x ** 2) for x in geoterm_xent))
    angle = T.arccos(-sum(T.sum(d * g) for d, g in zip(dottheta, geoterm_xent)) / norm1 / norm2) / np.pi * 180


    # Gradient descent update
    # updates = [(param, param - gparam) for param, gparam in zip(ae.params, gparams)]

    #   if not args.geo:
    #       updates = [(param, param - dg) for param, dg in zip(ae.params, dottheta)]
    #   else:
    #       updates = [(param, param - t1 + args.ratio * t2) for param, t1, t2 in zip(ae.params, dottheta, geoterm_xent)]
    def update_params(rate, weights, dottheta, geoterm_xent):
        if not args.geo:
            for i in range(len(ae.params)):
                ae.params[i].set_value(weights[i] - dottheta[i] * rate)
        else:
            for i in range(len(ae.params)):
                ae.params[i].set_value(weights[i] - rate * dottheta[i] + rate * args.ratio * geoterm_xent[i])


    # Initialize conjugate gradient with the previous result.
    if not args.geo:
        train = theano.function([index] + dottheta0 + geoterm0,
                                [reconstruction_loss, cost, all_ll, norm1, norm2, angle, gradXdottheta,
                                 dotthetaXGXdottheta] + dottheta + geoterm_xent,
                                givens=givens)
    else:
        train = theano.function([index] + dottheta0 + geoterm0,
                                [reconstruction_loss, cost, all_ll, norm1, norm2, angle, gradXp,
                                 pXGp] + dottheta + geoterm_xent,
                                givens=givens)

    if args.geo:
        log = open(
            'results/log_geo_lr_' + str(args.lr) + '_std_' + str(args.std) + '_ratio_' + str(args.ratio) + '.txt', 'w')
    else:
        log = open(
            'results/log_ng_lr_' + str(args.lr) + '_std_' + str(args.std) + '_ratio_' + str(args.ratio) + '.txt', 'w')

    for epoch in range(50):
        for i in range(n_batches):
            if i == 0 and epoch == 0:
                dt0 = [np.zeros_like(x.get_value()) for x in ae.params]
                gt0 = dt0[:]
                rloss, res, oldll, n1, n2, ang, gradp, pGp, *d = train(i, *(dt0 + gt0))
            else:
                decayed_d = [x * decay for x in d]
                rloss, res, oldll, n1, n2, ang, gradp, pGp, *d = train(i, *decayed_d)

            log.write(
                "train on batch %d, cost: %f, loss: %f, all_ll: %f, norm1: %f, norm2: %f, angle:%f\n" % (
                    i, res, rloss, oldll, n1, math.fabs(args.ratio) * n2, ang))
            log.flush()
            print("train on batch %d, cost: %f, loss: %f, all_ll: %f, norm1: %f, norm2: %f, angle: %f" % (
                i, res, rloss, oldll, n1, math.fabs(args.ratio) * n2, ang))

            dtheta = d[:len(ae.params)]
            gxent = d[len(ae.params):]
            weights = [x.get_value() for x in ae.params]
            rate = 1.0

            oldll = ae.compute_all_loss(train_set_x_np)
            update_params(rate, weights, dtheta, gxent)
            ll = ae.compute_all_loss(train_set_x_np)

            denom = (0.5 * pGp) - gradp

            rho = (oldll - ll) / denom
            if oldll > ll:
                rho = float('-Inf')

            print('oldll:%f, rho: %f, gradp: %f, pGp: %f' % (oldll, rho, gradp, pGp))
            log.write('oldll:%f, rho: %f, gradp: %f, pGp: %f\n' % (oldll, rho, gradp, pGp))

            if rho < 0.25 or math.isnan(rho):
                damp.set_value(np.asarray(damp.get_value() * 3.0 / 2.0, dtype=theano.config.floatX))
            elif rho > 0.75:
                damp.set_value(np.asarray(damp.get_value() * 2.0 / 3.0, dtype=theano.config.floatX))

            print('new damp: %f' % damp.get_value())
            log.write('new damp: %f\n' % damp.get_value())

            c = 10 ** (-2)
            j = 0
            while j < 60:
                if ll >= oldll + c * rate * gradp:
                    break
                else:
                    rate = 0.8 * rate
                    j += 1
                update_params(rate, weights, dtheta, gxent)
                ll = ae.compute_all_loss(train_set_x_np)
                print('%dth ll: %f' % (j, ll))
                log.write('%dth ll: %f\n' % (j, ll))

            print('rate: %f' % rate)
            log.write('rate: %f\n' % rate)

    log.write('-----------------------\n')
    print('----------------------')
    loss = 0.0
    for i in range(n_batches):
        rloss, res = test_error(i)
        loss += rloss
        print('test on batch %d, cost: %f, loss: %f' % (i, res, rloss))
        log.write('test on batch %d, cost: %f, loss: %f\n' % (i, res, rloss))
        log.flush()
    loss /= n_batches
    print('average training loss: %f' % (loss,))
    log.write('average training loss: %f\n' % (loss,))
    log.close()
