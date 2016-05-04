class instanceException(Exception):
    '''
    Represents no instance is available or anything relevant to insance
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class predictException(Exception):
    '''
    Indicate some error has occured in the process of training
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class interpreterException(Exception):
    '''
    Indicate something is wrong with interpreter
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class formatException(Exception):
    '''
    Indicate reader formate error
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class mlpException(Exception):
    '''
    Indicate mlp creation error
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class connectException(Exception):
    '''
    Indicate that there are some errors in layer-connections
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class shapeError(Exception):
    '''
    Indicate that there are some errors in layer input or output shapes
    '''
    def __init__(self,layer ,*args, **kwargs):
        self.exceptLayer = layer
        Exception.__init__(self, args, kwargs)

    def get_exceptLayer(self):
        return self.exceptLayer
