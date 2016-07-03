import numpy
try:
    import pylab
except ImportError:
    print ("pylab isn't available. If you use its functionality, it will crash.")
    print("It can be installed with 'pip install -q Pillow'")

import theano
import theano.tensor as T
from ether.component.init import init_shared, init_input
from ether.component.init import theano_rng


def build_rbm(v, W, bv, bh, k):
    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = theano_rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = theano_rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


from core import supervisedModel
class RnnRbm(supervisedModel):
    def __init__(self, n_visible, n_hidden, n_hidden_recurrent, **kwargs):
        '''
        Constructs and compiles Theano functions for training and sequence generation.
        :param n_hidden: Number of hidden units of the conditional RBMs.
        :param n_hidden_recurrent: Number of hidden units of the RNN.
        :param lr: Learning rate
        '''
        self.input = T.matrix()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_hidden_recurrent = n_hidden_recurrent

        if not isinstance(kwargs['weight'], dict):
            self.weight = kwargs['weight']
        else:
            assert not kwargs['weight'].has_key('shape')
            self.weight = init_shared(shape=(n_visible, n_hidden), **kwargs['weight'] )

        if not isinstance(kwargs['vbias'], dict):
            self.vbias = kwargs['vbias']
        else:
            assert not kwargs['vbais'].has_key('shape')
            self.vbias = init_shared(shape=(n_visible,), **kwargs['vbias'] )

        if not isinstance(kwargs['hbias'], dict):
            self.hbias = kwargs['hbias']
        else:
            assert not kwargs['hbais'].has_key('shape')
            self.hbias = init_shared(shape=(n_hidden,), **kwargs['hbias'] )

        if not isinstance(kwargs['ubias'], dict):
            self.ubias = kwargs['ubias']
        else:
            assert not kwargs['ubais'].has_key('shape')
            self.ubias = init_shared(shape=(n_hidden_recurrent,), **kwargs['ubias'] )

        if not isinstance(kwargs['uhweight'], dict):
            self.uhweight = kwargs['uhweight']
        else:
            assert not kwargs['uhweight'].has_key('shape')
            self.uhweight = init_shared(shape=(n_hidden_recurrent, n_hidden), **kwargs['uhweight'] )

        if not isinstance(kwargs['uvweight'], dict):
            self.uvweight= kwargs['uvweight']
        else:
            assert not kwargs['uvweight'].has_key('shape')
            self.uvweight = init_shared(shape=(n_hidden_recurrent, n_visible), **kwargs['uvweight'] )

        if not isinstance(kwargs['vuweight'], dict):
            self.vuweight = kwargs['vuweight']
        else:
            assert not kwargs['vuweight'].has_key('shape')
            self.vuweight = init_shared(shape=(n_visible, n_hidden_recurrent), **kwargs['vuweight'] )

        if not isinstance(kwargs['uuweight'], dict):
            self.uuweight = kwargs['uuweight']
        else:
            assert not kwargs['uuweight'].has_key('shape')
            self.uuweight = init_shared(shape=(n_hidden_recurrent, n_hidden_recurrent), **kwargs['uuweight'] )

        self.params = self.weight, self.vbias, self.hbias, self.uhweight, self.uvweight, self.vuweight, self.uuweight, self.ubias# learned parameters as shared

        u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden

        (u_t, bv_t, bh_t), updates_train = theano.scan(
            lambda v_t, u_tm1, *_: self.recurrence(v_t, u_tm1),
            sequences=self.input, outputs_info=[u0, None, None], non_sequences=params)
        v_sample, cost, monitor, updates_rbm = build_rbm(self.input, self.weight, bv_t[:], bh_t[:],
                                                         k=15)
        updates_train.update(updates_rbm)

        # symbolic loop for sequence generation
        (v_t, u_t), updates_generate = theano.scan(
            lambda u_tm1, *_: self.recurrence(None, u_tm1),
            outputs_info=[None, u0], non_sequences=self.params, n_steps=200)

        gradient = T.grad(cost, self.params, consider_constant=[v_sample])
        self.train_function = theano.function(
            [self.input],
            monitor,
            updates=updates_train
        )
        self.generate_function = theano.function(
            [],
            v_t,
            updates=updates_generate
        )

    def recurrence(self, v_t, u_tm1):
        bv_t = self.vbias + T.dot(u_tm1, self.uvweight)
        bh_t = self.hbias + T.dot(u_tm1, self.uhweight)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((self.n_visible,)), self.weight, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(self.ubias+ T.dot(v_t, self.vuweight) + T.dot(u_tm1, self.uuweight))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    def feed_forward(self, input):
        #TODO
        sample = self.generate_function()
        return sample
