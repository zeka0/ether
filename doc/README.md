##Version-Notes
The current version is radically different from previous version.
First of all, I must say sorry to the interface-changes all around the library.
It's because I have to accomondate new models, i.e, RBM into the scope of the library.

##Introduction
- __Introduction of packages__ are in:
  1. `\instance.md`
  2. `\layer.md`
  3. `model.md`
  4. `\trainier.md`
  5. `\pickle.md`
- __Notes for using python__ is in `\lang.md`

##Pre-warning
Some part of the library can be very tricky.
So pay close attention when you want to learn from it.
I take all the blame for not making the library easy to use.
And with that goal in my mind, I am dedicated to make it better.

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

###Parameters of Layers and Components
An advanced version of parameters are offered.
Since the __T.grad__ has a extra parameter called _considered\_constant_, it's extremly important in the case of computing gradients.
Moreover, some components may require to compute extra _updates_ in the _training process_.
