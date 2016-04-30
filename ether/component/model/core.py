class model(object):
    def get_cost(self):
        '''
        :return the cost to minimize
        '''
        raise NotImplementedError()

    def get_monitoring_cost(self):
        raise NotImplementedError()

    def get_extra_updates(self):
        '''
        In some models, they need to update their inner state.
        This is done by using this method.
        '''
        raise NotImplementedError()

    def get_outputTensor(self):
        '''
        This is needed to provide the ability to link with other models
        '''
        raise NotImplementedError()

    def get_gparams(self):
        '''
        :return a list of tuples of (grad, para),
        This utility is need because some graph structures may have optimized algorithnms.
        '''
        raise NotImplementedError()

    def get_params(self):
        '''
        :return trainable parameters of this model.
        '''
        raise NotImplementedError()

    def compile(self):
        '''
        This method is called every-time before calling the get_gparams or other functions.
        In short, this function is used to compute the necessary parts of the cost, grads or updates.
        '''
        pass

    def get_inputTensor(self):
        '''
        :return input-tensor of the model
        '''
        raise NotImplementedError()