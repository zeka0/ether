#Author - Alphasis Zeka
##Whose English Name is Ether Wei
__OMG I have finally built my first ever lenet!!!__

###Pre-note
- Docs can be found in `\doc`
- The files in the directory `\deprecated` are from older versions.

###Introduction:
- It's a __theano__-based neural-network.
- Required modulas include: __numpy__, __theano__.
- It's a little bit complicated to learn to use at first.
- Before you start using the nnet, plz read the __Rules & Cautions__ blocks below.

###Version notes
In this version, I have replaced the old _signal.conv2d_. 
Because though this version provides you with more flexibilities, it's considerably slower than _T.nnet.conv2d_.
And also, I have restrained the rules.

###Future Improvements
- [ ] RNN
- [ ] Restricted Boltzmann Machine
- [ ] Auto-encoder
- [ ] Sparse coding


##Notes From Author:
- Neural network is really tough to learn.
- The key part in bulding a nnet is in __selecting proper parameters for layers__.
- And also, if time premitted, grabing a book about numpy is a really good choice.
- I used the `\clearPyc.py` to clean the pyc files created by python interpreter before pushing to git.

##Cautions
- It's strongly recommended to read the `doc` before using this project.
- If you have never touched __numpy__ or __theano__ before, I suggest you to try them out yourself a little bit.
It's because sometimes when you find a bug, having some knowledge of them can help great lot.
- __Global variables in python are tricky to use.__ You should avoid them as early as possible.
However, just like their counterparts in cpp, you can __provide scope for the global variables to operate.__
This can be done by importing only the modula instead of all the contents of the modula and use the syntax:
```python
import core
core.root_dir = 'C:\'
```

##Rules
- The input to the nnet should contain a dimension represent the _batch size_.
- The __mini-batch__ isn't supported.
However, you may see some implementations of the layers support the mini-batch, it's not supported in the optimizer.
Thought this could mean that the computation could be slower, the mini-batch can be simulated anyway.
