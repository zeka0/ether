from ether.component.core import component
class model(component):
    def get_cost(self):
        '''
        :return the cost to minimize
        :rtype theano.tensor
        '''
        raise NotImplementedError()

    def get_extra_updates(self):
        '''
        In some models, they need to update their inner state.
        This is done by using this method.
        :return extra updates in optimizing phase of the cost of the model
        :rtype dict
        '''
        raise NotImplementedError()

    def get_gparams(self):
        '''
        :return a list of tuples of (grad, para),
        This utility is need because some graph structures may have optimized algorithnms.
        '''
        raise NotImplementedError()

    def feed_forward(self, input):
        '''
        Inplemented by the model
        '''
        raise NotImplementedError()

class unsupervisedModel(model):
    def get_monitoring_cost(self):
        '''
        :rtype theano.tensor
        '''
        raise NotImplementedError()

class supervisedModel(model):
    def get_targetTensor(self):
        '''
        :return the symbolic tensor which represents the target of the model
        :rtype theano.tensor
        '''
        raise NotImplementedError()
