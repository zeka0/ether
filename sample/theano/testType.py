from theano import tensor as T
import numpy as np
from numpy.matlib import *

x = T.dscalar()
v = T.vector()
z = v.sum()
print z.type == T.dscalar

x1 = T.dscalar()
x2 = T.dscalar()
x3 = T.dscalar()
zz = T.stack(x1, x2, x3)
print zz.type == T.vector
print zz.ndim
print zz.shape
