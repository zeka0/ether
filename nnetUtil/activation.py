__author__ = 'mac'

import theano
from theano import tensor as T

#TODO: bugs found using this modula
def tanh(x):
    return T.tanh(x)

def softmax(x):
    return T.nnet.softmax

def softplus(x):
    return T.nnet.softplus

def sigmoid(x):
    return T.nnet.sigmoid

def relu(x, border=0):
    return T.clip(x, border, x)

def clip(x, min, max):
    return T.clip(x, min, max)

def max_out(x):
    return T.max_and_argmax(x)
