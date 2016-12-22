import theano
from theano import tensor as T

x = T.scalar()
def iterate(x):
    x = x * 2
    return x

out, updates = theano.scan(fn=iterate, outputs_info=x, n_steps=2)
fun = theano.function(inputs=[x], outputs=out)
print fun(1)
