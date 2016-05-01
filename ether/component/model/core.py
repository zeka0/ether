from ether.component.core import component
class model(component):
    #TODO add get_inputs(self) to return the required inputs to the model
    #TODO still the targetTensor is a problem
    #TODO different models required different inputs
    def get_cost(self):
        '''
        :return the cost to minimize
        :rtype theano.tensor
        '''
        raise NotImplementedError()

    def get_monitoring_cost(self):
        '''
        :rtype theano.tensor
        '''
        raise NotImplementedError()

    def get_extra_updates(self):
        '''
        In some models, they need to update their inner state.
        This is done by using this method.
        :rtype list
        '''
        raise NotImplementedError()

    def get_gparams(self):
        '''
        :return a list of tuples of (grad, para),
        This utility is need because some graph structures may have optimized algorithnms.
        '''
        raise NotImplementedError()

    def compile(self):
        '''
        This method is called every-time before calling the get_gparams or other functions.
        In short, this function is used to compute the necessary parts of the cost, grads or updates.
        '''
        pass
