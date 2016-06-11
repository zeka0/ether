import theano
from ae import autoEncoder

class denoisingAutoEncoder(autoEncoder):
    def __init__(self, n_visible, n_hidden, corruption_level, **kwargs):
        self.corruption_level = corruption_level
        autoEncoder.__init__(self, n_visible, n_hidden, **kwargs)

    def get_outputTensor(self):
        tilde_x = self.get_corrupted_input(self.input, self.corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        return z

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

