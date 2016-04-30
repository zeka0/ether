import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from core import model

class RestrictedBM(model):
    def __init__(self,
                 n_visible, n_hidden,
                 numpy_rng=None, theano_rng=None,
                 **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if kwargs.has_key('weight'):
            self.W = kwargs['weight']
        else:
            self.W = theano.shared( np.asarray( numpy_rng.uniform(
                low=-4 * np.sqrt( 6./ (n_hidden + n_visible) ),
                high=4 * np.sqrt( 6./ (n_hidden + n_visible) ),
                size=(n_visible, n_hidden)
            ) ) )

        if kwargs.has_key('hbias'):
            self.hbias = kwargs['hbias']
        else:
            self.hbias = theano.shared( value=np.zeros(n_hidden, dtype=np.float), borrow=True )

        if kwargs.has_key('vbias'):
            self.vbias = kwargs['vbias']
        else:
            self.vbias = theano.shared( value=np.zeros( n_visible, dtype=np.float ), borrow=True )

        if kwargs.has_key('input'):
            self.input = kwargs['input']
        else:
            self.input = T.matrix('input')
        self.theano_rng = theano_rng

    def get_params(self):
        return [self.W, self.vbias, self.hbias]

    def prop_up(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.prop_up(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def prop_down(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.prop_down(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -vbias_term - hidden_term

    def scan_nhv(self, chain_start, k=1):
        ([
             self.pre_sigmoid_nvs,
             self.nv_means,
             self.nv_samples,
             self.pre_sigmoid_nhs,
             self.nh_means,
             self.nh_samples
         ], updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name='gibbs_hvh'
        )

    def get_cost(self, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        self.scan_nhv(chain_start, k)
        chain_end = self.nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        self.updates = dict()
        return cost

    def get_monitoring_cost(self, persistent=None, k=1):
        if persistent:
            self.updates[persistent] = self.nh_samples[-1]
            cost = self.get_pseudo_likelihood_cost()
        else:
            cost = self.get_reconstruction_cost(self.pre_sigmoid_nvs[-1])
        return cost


    def get_pseudo_likelihood_cost(self):
        bit_i_idx = theano.shared(value=0)
        xi = T,round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        self.updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def get_grad_para_tuple(self, cost, chain_end):
        gparams = T.grad(cost, self.get_params(), consider_constant=[chain_end])
        gpts = []
        for gparam, para in zip(gparams, self.get_params()):
            gpts += (gparam, para)
        return gpts

    def get_extra_updates(self):
        return self.updates
