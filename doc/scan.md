#Analysis of theano.scan

###Introduction
I personally found that theano.scan is the most astonishing method provided by the theano library.

###Params
- __output_info__ the initial values of the parameters
- __sequences__ the sequences for the scan to iterate
- __non_sequences__ the constant in the parameters
- __n_steps__ the number of iterations, may conflicts with **sequences**

###The order of the parameters
The general order of parameters applied to __fn(the first parameter of theano.scan)__ is as follows.
- First is the __output_info__.
- Second is the __sequences__ which can be a list or just a tensor. If sequences is a list, then during iteration i, the arguements applied
to fn is as follows: __pseq[i] for pseq in sequences__
- Last is the __non_sequences__, they are simply passed to the arguement.

###Special treatment for output_info
Unlike __sequences and non_sequences__, __output_info__ is an exception, and that's why scan is so powerful.
If you can only use __sequences and non_sequences__, you need to provide arguements for the parameters of fn at every time step.
So, you can't convert the following code into scan:
```python
x = 1
for i in xrange(10):
    x = x * 3
```
That's where __output_info__ comes handy.
The actual rule of output_info not only comes as the initial values of the parameters of the fn,
but also as __place holders__ to tell scan that at these locations, the parameters are replaced.
For example, a function receives fn(A, B, C) and returns (A*2, B, C).
If output_info = A, then in first iteration scan passes (A, B, C) to fn.
In the second iteration however, scan uses the-first-iteration-of-fn's return _first value as the first arguement_ for fn.
So the above for-loop can be rewritten as:
```python
def multiply_by_three(x):
    return x*3

x = T.dscalar()
scan(fn=multiply_by_three, output_info=x, n_step=10)
```

###Notes for writting Scan
- __The order of parameters of fn is very important__
- __The number of the returned values of fn should be greater or equal than the len(output_info)__.
This is because of the output_info here.
- __len(output_info) + len(sequences) + len(non_sequences) == len(parameters of fn)__
- __If you really can't decide the number of parameters or returned values, use '*args'__.
Be sure to follow the tradition to use the symbol '**_**' as place holders.
