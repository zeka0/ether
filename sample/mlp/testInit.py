from nnet.mlp.initialize import *

x = normal.init_filters(1, (5,5))
for y in x:
    print y.get_value()