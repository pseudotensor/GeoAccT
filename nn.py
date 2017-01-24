"""
neural network layers and helper functions
"""

import theano
import theano.tensor as T
import numpy as np
import random


class FullyConnectedLayer(object):
    # Use spare initialization as Martens, 2010.
    def __init__(self, input, n_in, n_out, std=1.0, W=None, b=None, activation=T.nnet.sigmoid):
        self.input = input
        self.activation = activation
        if b is None:
            self.b = theano.shared(
                np.zeros((n_out,), dtype=theano.config.floatX),
                borrow=True
            )
        else:
            self.b = b
        if W is None:
            numconn = 15
            W0 = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            for i in range(n_out):
                connected = random.sample(range(n_in), numconn)
                W0[connected, i] = np.asarray(
                    np.random.normal(
                        loc=0.0,
                        scale=std,
                        size=(numconn,)
                    ), dtype=theano.config.floatX
                )
            self.W = theano.shared(W0, borrow=True)
        # self.W = theano.shared(
        #               np.asarray(
        #                   np.random.normal(
        #                       loc=0.0,
        #                       scale=std,
        #                       size=(n_in, n_out)
        #                   ), dtype=theano.config.floatX
        #               )
        #           )
        else:
            self.W = W
        self.raw_output = T.dot(self.input, self.W) + self.b
        self.output = self.activation(T.dot(self.input, self.W) + self.b)
        self.params = [self.W, self.b]


class AutoEncoder(object):
    def __init__(self, input, all_x, size, batchsize, datasize, std=1.0, weight=None, activation=T.nnet.sigmoid):
        """
        Create an autoencoder by stacking FullyConnectedLayers
        :param input: size must be #samples * #dimension
        :param size: list of numer of neurons of each layer
        :param weight: stored weights if any
        """
        self.input = input
        self.all_x = all_x
        self.size = size
        self.activation = activation
        self.layers = []
        self.batchsize = batchsize
        self.datasize = datasize
        # create input layer
        input_layer = FullyConnectedLayer(self.input, size[0], size[1], std=std,
                                          activation=activation) if weight is None else \
            FullyConnectedLayer(self.input, size[0], size[1], std=std, W=weight[0], b=weight[1],
                                activation=activation)
        self.layers.append(input_layer)
        self.params = input_layer.params

        encode_i = len(size) // 2

        for i in range(1, len(size) - 1):
            input = self.layers[-1].output

            if i != encode_i - 1:
                new_layer = FullyConnectedLayer(input, size[i], size[i + 1], std=std,
                                                activation=activation) if weight is None else \
                    FullyConnectedLayer(input, size[i], size[i + 1], std=std, W=weight[i * 2], b=weight[i * 2 + 1],
                                        activation=activation)
            else:
                new_layer = FullyConnectedLayer(input, size[i], size[i + 1], std=std,
                                                activation=lambda x: x) if weight is None else \
                    FullyConnectedLayer(input, size[i], size[i + 1], std=std, W=weight[i * 2], b=weight[i * 2 + 1],
                                        activation=lambda x: x)
            self.layers.append(new_layer)
            self.params += new_layer.params

        self.raw_output = self.layers[-1].raw_output
        self.output = self.layers[-1].output

    def get_cost(self, label):
        # TODO: only use cross entropy here. Need to support more
        # Note the annoying case that 0*log(0) = NaN
        # Do not use this: return -T.sum(label * T.log(self.output) + (1 - label) * T.log(1 - self.output)) / float(self.batchsize)
        # The following is different from the ``more stable'' version of Martens 2010.
        return T.sum(self.raw_output * (1.0 - label) + T.log(1.0 + T.exp(-self.raw_output))) / float(self.batchsize)

    def get_reconstruction_loss(self, label):
        return T.sum((self.output - label) ** 2) / float(self.batchsize)

    def backprop(self, label):
        l = len(self.size)
        Db = [None] * l
        Db2 = [None] * l
        DW = [None] * l
        DW2 = [None] * l
        Ds = [None] * l
        Da = [None] * l
        Ds[l - 1] = T.grad(self.get_cost(label), self.layers[-1].raw_output)
        a = [self.input] + [self.layers[i].output for i in range(l - 1)]

        encode_i = len(self.size) // 2
        for i in range(l - 1, 0, -1):
            # TODO: Only supports logistic activation
            if i != l - 1:
                if i != encode_i:
                    Ds[i] = Da[i] * a[i] * (1 - a[i])
                else:
                    Ds[i] = Da[i]
            DW[i] = T.dot(a[i - 1].T, Ds[i])
            DW2[i] = T.dot(T.square(a[i - 1].T), T.square(Ds[i]))
            Db[i] = T.sum(Ds[i], axis=0)
            Db2[i] = T.sum(T.square(Ds[i]), axis=0)
            Da[i - 1] = T.dot(Ds[i], self.layers[i - 1].W.T)

        return (list(sum(zip(DW, Db), ()))[2:], list(sum(zip(DW2, Db2), ()))[2:])

    def backprop_all(self, alpha):
        l = len(self.size)
        data_size = self.datasize
        chucks = data_size // self.batchsize
        assert (data_size % self.batchsize == 0)
        DW_ave = [0] + [T.zeros_like(self.layers[i].W) for i in range(l - 1)]
        Db_ave = [0] + [T.zeros_like(self.layers[i].b) for i in range(l - 1)]
        DW2_ave = [0] + [T.zeros_like(self.layers[i].W) for i in range(l - 1)]
        Db2_ave = [0] + [T.zeros_like(self.layers[i].b) for i in range(l - 1)]
        ll = T.zeros((), dtype=theano.config.floatX)
        for I in range(chucks):
            input = self.all_x[I * self.batchsize: (I + 1) * self.batchsize, ]
            a = [input] + [None] * (l - 1)
            s = [None] * l
            Da = [T.zeros_like(input)] + [None] * (l - 1)
            Ds = [None] * l
            DW = [0] + [T.zeros_like(self.layers[i].W) for i in range(l - 1)]
            Db = [0] + [T.zeros_like(self.layers[i].b) for i in range(l - 1)]
            DW2 = [0] + [T.zeros_like(self.layers[i].W) for i in range(l - 1)]
            Db2 = [0] + [T.zeros_like(self.layers[i].b) for i in range(l - 1)]
            encode_i = len(self.size) // 2
            for i in range(1, l):
                W = self.params[(i - 1) * 2]
                b = self.params[(i - 1) * 2 + 1]
                s[i] = T.dot(a[i - 1], W) + b
                if i != encode_i:
                    a[i] = 1.0 / (1.0 + T.exp(-s[i]))
                else:
                    a[i] = s[i]

            Ds[l - 1] = alpha * (a[-1] - input) / self.batchsize
            # ll += T.sum(s[-1] * (input - 1.0) - T.log(1.0 + T.exp(-s[-1]))) / float(self.batchsize)
            ll += T.sum(s[-1] * (input - (s[-1] >= 0)) - T.log(1.0 + T.exp(s[-1] - 2* s[-1] * (s[-1] >= 0)))) / float(self.batchsize)
            for i in range(l - 1, 0, -1):
                # TODO: Only supports logistic activation
                if i != l - 1:
                    if i != encode_i:
                        Ds[i] = Da[i] * a[i] * (1 - a[i])
                    else:
                        Ds[i] = Da[i]
                DW[i] = T.dot(a[i - 1].T, Ds[i])
                DW2[i] = T.dot(T.square(a[i - 1].T), T.square(Ds[i]))
                Db[i] = T.sum(Ds[i], axis=0)
                Db2[i] = T.sum(T.square(Ds[i]), axis=0)
                Da[i - 1] = T.dot(Ds[i], self.layers[i - 1].W.T)

            DW_ave = [x + y for x, y in zip(DW_ave, DW)]
            Db_ave = [x + y for x, y in zip(Db_ave, Db)]
            DW2_ave = [x + y for x, y in zip(DW2_ave, DW2)]
            Db2_ave = [x + y for x, y in zip(Db2_ave, Db2)]
        DW_ave = [x / chucks for x in DW_ave]
        Db_ave = [x / chucks for x in Db_ave]
        DW2_ave = [x / chucks for x in DW2_ave]
        Db2_ave = [x / chucks for x in Db2_ave]
        ll /= chucks
        return (list(sum(zip(DW_ave, Db_ave), ()))[2:], list(sum(zip(DW2_ave, Db2_ave), ()))[2:], ll)

    def compute_all_loss(self, train_set):
        l = len(self.size)
        data_size = self.datasize
        chucks = data_size // self.batchsize
        assert (data_size % self.batchsize == 0)
        ll = 0.0
        for I in range(chucks):
            input = train_set[I * self.batchsize: (I + 1) * self.batchsize, ]
            a = [input] + [None] * (l - 1)
            s = [None] * l
            encode_i = len(self.size) // 2
            for i in range(1, l):
                W = self.params[(i - 1) * 2].get_value()
                b = self.params[(i - 1) * 2 + 1].get_value()
                s[i] = np.dot(a[i - 1], W) + b
                if i != encode_i:
                    a[i] = 1.0 / (1.0 + np.exp(-s[i]))
                else:
                    a[i] = s[i]

            # ll += np.sum(s[-1] * (input - 1.0) - np.log(1.0 + np.exp(-s[-1]))) / float(self.batchsize)
            ll += np.sum(s[-1] * (input - (s[-1] >= 0)) - np.log(1.0 + np.exp(s[-1] - 2*s[-1]*(s[-1]>=0)))) / float(self.batchsize)

        ll /= chucks
        return ll

    def metric_product(self, v):
        """
        :param v: the structure must correspond to [W1 b1 W2 b2 ... ]
        :return: \sum_{i=1}^o \lambda_i \partial_\mu y_i \partial_\nu y_i v^\nu
        """
        l = len(self.size)
        a = [self.input] + [self.layers[i].output for i in range(l - 1)]
        Ra = [T.zeros_like(self.input)] + [None] * (l - 1)
        Rs = [None] * l
        Db = [None] * l
        DW = [None] * l
        Ds = [None] * l
        Da = [None] * l

        encode_i = len(self.size) // 2
        for i in range(1, l):
            RW_i = v[(i - 1) * 2]
            Rb_i = v[(i - 1) * 2 + 1]
            Rs[i] = T.dot(a[i - 1], RW_i) + T.dot(Ra[i - 1], self.layers[i - 1].W) + Rb_i
            if i != encode_i:
                Ra[i] = a[i] * (1 - a[i]) * Rs[i]
            else:
                Ra[i] = Rs[i]

        # Da[l - 1] = Ra[l - 1]  # TODO: squared loss only
        # Da[l - 1] = 1.0 / (self.output * (1 - self.output)) * Ra[l - 1] / self.batchsize  # TODO: cross-entropy only
        # The following change is critical for avoiding NaNs.
        Ds[l - 1] = Ra[l - 1] / self.batchsize  # TODO: cross-entropy only
        for i in range(l - 1, 0, -1):
            if i != l - 1:
                if i != encode_i:
                    Ds[i] = Da[i] * a[i] * (1 - a[i])
                else:
                    Ds[i] = Da[i]
            DW[i] = T.dot(a[i - 1].T, Ds[i])
            Db[i] = T.sum(Ds[i], axis=0)
            Da[i - 1] = T.dot(Ds[i], self.layers[i - 1].W.T)

        return list(sum(zip(DW, Db), ()))[2:]

    def term1(self, v):
        """
        Compute the 1st difficult term as in Algorithm 5
        :param v: \dot \theta
        :return: the result
        """
        l = len(self.size)
        a = [self.input] + [self.layers[i].output for i in range(l - 1)]
        Ra = [T.zeros_like(self.input)] + [None] * (l - 1)
        Sa = [T.zeros_like(self.input)] + [None] * (l - 1)
        Rs = [None] * l
        Ss = [None] * l
        DW = [None] * l
        Ds = [None] * l
        Db = [None] * l
        Da = [None] * l

        encode_i = len(self.size) // 2
        for i in range(1, l):
            RW_i = v[(i - 1) * 2]
            Rb_i = v[(i - 1) * 2 + 1]
            W_i = self.layers[i - 1].W
            Rs[i] = T.dot(a[i - 1], RW_i) + T.dot(Ra[i - 1], W_i) + Rb_i
            if i != encode_i:
                Ra[i] = a[i] * (1 - a[i]) * Rs[i]
            else:
                Ra[i] = Rs[i]
            Ss[i] = 2 * T.dot(Ra[i - 1], RW_i) + T.dot(Sa[i - 1], W_i)
            if i != encode_i:
                Sa[i] = (2 * a[i] - 1) * a[i] * (a[i] - 1) * Rs[i] * Rs[i] + a[i] * (1 - a[i]) * Ss[i]
            else:
                Sa[i] = Ss[i]

        # Da[l - 1] = Sa[l - 1]  # TODO: squared loss only
        Ds[l - 1] = Sa[l - 1] / self.batchsize  # TODO: cross-entropy only

        for i in range(l - 1, 0, -1):
            if i != l - 1:
                if i != encode_i:
                    Ds[i] = Da[i] * a[i] * (1 - a[i])
                else:
                    Ds[i] = Da[i]
            DW[i] = T.dot(a[i - 1].T, Ds[i])
            Db[i] = T.sum(Ds[i], axis=0)
            Da[i - 1] = T.dot(Ds[i], self.layers[i - 1].W.T)

        return list(sum(zip(DW, Db), ()))[2:]

    def term2(self, v):
        """
        Compute the 2st difficult term as in Algorithm 6
        :param v: \dot \theta
        :return: the result
        """
        l = len(self.size)
        a = [self.input] + [self.layers[i].output for i in range(l - 1)]
        Ra = [T.zeros_like(self.input)] + [None] * (l - 1)
        Rs = [None] * l
        DW = [None] * l
        Ds = [None] * l
        Db = [None] * l
        Da = [None] * l

        encode_i = len(self.size) // 2
        for i in range(1, l):
            RW_i = v[(i - 1) * 2]
            Rb_i = v[(i - 1) * 2 + 1]
            W_i = self.layers[i - 1].W
            Rs[i] = T.dot(a[i - 1], RW_i) + T.dot(Ra[i - 1], W_i) + Rb_i
            if i != encode_i:
                Ra[i] = a[i] * (1 - a[i]) * Rs[i]
            else:
                Ra[i] = Rs[i]

        # Da[l - 1] = Ra[l - 1] ** 2  # TODO: suqared loss only
        # TODO: cross-entropy only
        Ds[l - 1] = ((2.0 * self.output - 1.0) / 2.0 / (self.output * (1 - self.output))) * Ra[
                                                                                                l - 1] ** 2 / self.batchsize

        for i in range(l - 1, 0, -1):
            if i != l - 1:
                if i != encode_i:
                    Ds[i] = Da[i] * a[i] * (1 - a[i])
                else:
                    Ds[i] = Da[i]
            DW[i] = T.dot(a[i - 1].T, Ds[i])
            Db[i] = T.sum(Ds[i], axis=0)
            Da[i - 1] = T.dot(Ds[i], self.layers[i - 1].W.T)

        return list(sum(zip(DW, Db), ()))[2:]
