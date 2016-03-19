__author__ = 'mac'

class instanceException(Exception):
    '''
    Represents no insance is available or anything relevant to insance
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
    Indecate something is wrong with interpreter
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class formatException(Exception):
    '''
    Indicate reader formating error
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)

class MLPException(Exception):
    '''
    Indicate MLP creation error
    '''
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, args, kwargs)
