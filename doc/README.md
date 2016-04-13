##Introduction
- __Introduction of packages__ are in:
  1. `\instance.md`
  2. `\layer.md`
  3. `lenet5.md`
  4. `\trainier.md`
  5. `\pickle.md`
- __Notes for using python__ is in `\lang.md`

##Pre-warning

###Shape mechanism
The present version of the nnet uses a rather strongly checked shape mechanisms.
That means it will check shape automaticly. However, some problems araise from this scheme.
One thing to remind you guys is that all the tensor here are of type __theano.tensor.matrix__.
It's for the sake of simplicity and robust. However, that also means that __you can't pass a tensor whose shape is not of 2 dims__.
So, a _3-dim-ndarray or a scala_ can't be directly added to the _inputLayer_.

###Order of layers
You will notice the the __nnet__ object requires you pass an array of layers to its constructor.
However, the order of the array is strictly checked.
First of all, _layers[0] must represent the inputLayer_.
Secondly, _layers[-1] must represent the outputLayer_.
In the current version of nnet, the order in the array other than the postion 0 and -1 are not checked.
But maintaing them in __front to back order__ is a good practice and is recommended.
