from theano import tensor as T
import theano
import numpy as np

def forward_prop_step(x_t, s_t_prev, U, V, W):
    s_t = T.nnet.softmax(x_t.dot(U) + s_t_prev.dot(W))
    o_t = T.dot(s_t, V)
    return [o_t, s_t]

U = theano.shared(np.ones((100,100)))
V = theano.shared(np.ones((100,100)))
W = theano.shared(np.ones((100,100)))

I = theano.shared(np.zeros((10,100)))

[o_t, s_t], updates = theano.scan(fn=forward_prop_step,
                                  sequences=I,
                                  outputs_info=[None, dict(initial=T.zeros((1, 100)))],
                                  non_sequences=[U, V, W],
                                  strict=True)
