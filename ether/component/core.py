class component:
    '''
    The general interface of model
    Designed to provide utilities to link between models or components
    '''
    def get_params(self):
        '''
        :return the trainable parameters in the component, return an empty list represent no trainable parameters.
        :rtype has to be list
        '''
        raise NotImplementedError()

    def get_nparams(self):
        '''
        Name-parameter-dict
        Used in self.__dict__.update(obj.get_nparams())
        Dedicated to sharing parameters across the components
        :return the name and the symbolic tensor of the parameters
        :rtype dictionary
        '''
        raise NotImplementedError()

    def get_inputTensor(self):
        '''
        :return the input to the component
        :rtype theano.tensor
        '''
        raise NotImplementedError()

    def get_outputTensor(self):
        '''
        :return the output of the component
        :rtype theano.tensor
        '''
        raise NotImplementedError()

    def get_inputShape(self):
        '''
        :return the shape of the input
        :rtype tuple
        '''
        raise NotImplementedError()

    def get_outputShape(self):
        '''
        :return the shape of the output
        :rtype tuple
        '''
        raise NotImplementedError()