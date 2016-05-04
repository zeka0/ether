import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from core import unsupervisedModel
from ether.component.init import init_shared, init_input

class RestrictedBM(unsupervisedModel):
    def __init__(self,
                 n_visible, n_hidden, train_k=1, sample_k=1000,
                 theano_rng=None, persistent=None,
                 **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if theano_rng is None:
            theano_rng = RandomStreams(np.random.randint(2**30))
        assert kwargs.has_key('weight')
        assert kwargs.has_key('hbias')
        assert kwargs.has_key('vbias')
        assert not kwargs['weight'].has_key('shape')
        assert not kwargs['vbias'].has_key('shape')
        assert not kwargs['hbias'].has_key('shape')

        self.W = init_shared(shape=(n_visible, n_hidden), **kwargs['weight'] )
        self.hbias = init_shared(shape=(n_hidden,), **kwargs['hbias'] )
        self.vbias = init_shared(shape=(n_visible,), **kwargs['vbias'] )
        self.theano_rng = theano_rng
        self.persistent = persistent
        self.train_k = train_k
        self.sample_k = sample_k
        self.input = T.matrix()

    def get_params(self):
        return [self.W, self.vbias, self.hbias]

    def prop_up(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.prop_up(v0_sample)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def prop_down(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.prop_down(h0_sample)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean, dtype=theano.config.floatX)
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

    def scan_nhv(self, chain_start):
        ([
             self.pre_sigmoid_nvs,
             self.nv_means,
             self.nv_samples,
             self.pre_sigmoid_nhs,
             self.nh_means,
             self.nh_samples
         ], self.updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=self.train_k,
            name='gibbs_hvh'
        )
        if self.persistent:
            self.updates[self.persistent] = self.nh_samples[-1]

    def get_cost(self):
        if not hasattr(self, 'cost'):
            pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
            if self.persistent is None:
                chain_start = ph_sample
            else:
                chain_start = self.persistent
            self.scan_nhv(chain_start)
            self.chain_end = self.nv_samples[-1]
            self.cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(self.chain_end))
        return self.cost

    def get_monitoring_cost(self):
        if not hasattr(self, 'monitor_cost'):
            if self.persistent:
                self.monitor_cost = self.get_pseudo_likelihood_cost()
            else:
                self.monitor_cost = self.get_reconstruction_cost(self.pre_sigmoid_nvs[-1])
        return self.monitor_cost

    def get_pseudo_likelihood_cost(self):
        bit_i_idx = theano.shared(value=0)
        xi = T.round(self.input)
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

    def get_gparams(self):
        cost = self.get_cost()
        gparams = T.grad(cost, self.get_params(), consider_constant=[self.chain_end])
        gpts = []
        for gparam, para in zip(gparams, self.get_params()):
            gpts.append((gparam, para))
        return gpts

    def get_extra_updates(self):
        return self.updates

    def get_outputTensor(self):
        raise NotImplementedError('No specific output can be caculated')

    def get_inputTensor(self):
        return self.input

    def get_inputShape(self):
        return (self.n_visible, 1)

    def get_outputShape(self):
        return (self.n_visible, 1)

    def reset_sample_k(self, sample_k):
        self.sample_k = sample_k

    def feed_forward(self, input):
        if not hasattr(self, 'sample_fn'):
            self.compile_sample_fn()
        self.persistent_vis_chain.set_value(input)
        vis_mf, vis_sample = self.sample_fn()
        return vis_mf

    def compile_sample_fn(self):
        self.persistent_vis_chain = theano.shared( np.zeros(self.get_inputShape()) )
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, self.persistent_vis_chain],
            n_steps=self.sample_k,
            name="gibbs_vhv"
        )
        updates.update({self.persistent_vis_chain: vis_samples[-1]})
        self.sample_fn = theano.function(
            [],
            [
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates
        )
