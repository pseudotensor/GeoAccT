import gzip, pickle, theano
import theano.tensor as T
import numpy as np

def load(name):
    """
    load dataset and turn it to shared variables
    :param name: the name of dataset used
    :return: shared pairs of train, valid, and test
    """
    assert (name == 'MNIST')
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    f.close()
    train_set_x_np, train_set_y_np = train_set
    train_set_x_np = np.asarray(train_set_x_np, dtype=theano.config.floatX)
    train_set_y_np = np.asarray(train_set_y_np, dtype=theano.config.floatX)
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, train_set_x_np, train_set_y_np


def shared_dataset(data):
    x, y = data
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')
