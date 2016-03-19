import sample.theano
from sample.theano import tensor as T
import numpy as np
from numpy.matlib import *

input = rand( (1, 12) )
x = rand(12)
print input
print input.shape
print x
print x.shape
y = 1
print len( y )

