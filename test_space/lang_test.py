import numpy as np

class XTest(object):
    def __init__(self):
        print self.__class__
        print self.__module__
        print self.__str__()
        print self.__class__.__name__


x = XTest()
