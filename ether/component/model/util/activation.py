import theano
from theano import tensor as T

def tanh(x):
    return T.tanh(x)

def softmax(x):
    return T.nnet.softmax(x)

def softplus(x):
    return T.nnet.softplus(x)

def sigmoid(x):
    return T.nnet.sigmoid(x)

def relu(x, border=0):
    return T.clip(x, border, x)

def clip(x, min, max):
    return T.clip(x, min, max)

def max_out(x):
    return T.max_and_argmax(x)

def argmax(x):
    return T.argmax(x)
