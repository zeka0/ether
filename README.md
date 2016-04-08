##Author - Alphasis Zeka

##Introduction:
-It's a __theano__-based neural-network.
-Required modulas include: __numpy__, __theano__.
-It's a little bit complicated to learn to use at first.

##News:
-__Good News:__The reborn nnet has now returned! The new version contained a brand-new conv2DLayer.
-__Bad News:__

##Future Improvements

##Warnings
-In numpy, calling ndarray constructor directly may cause problems. It's advised to call array instead.
-In __conv2DLayer__ module, I used __signal.conv2d__ instead of __nnet.conv2d__.

##Quick Start
First, you should establish a __database__ to accommodate the __instance__. Then you can build a __validator__(used to report the error rates) and a
__optimizer__(used to train the nnet). After that, you can start creating objects of __layer__ classes. __Connect__ them when nessary to create a __nnet__.
Examples can be seen in `\test\lenet5\lenet5.py`.

##Notes From Author:
-Neural network is really tough to learn.
-And also, if time premitted, grabing a book about numpy is a really good choice.
-__Yoshua Bengio__ is a pretty nice guy(But his books are not friendly at all).
-I used the `\test\clearPyc.py` to clean the pyc files created by python interpreter before pushing to git.

##BookNotes
use
'''python
def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

    if hasattr(self, 'attrName'):
        pass
'''
to add the attributes into the object
