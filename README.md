#Author - Ether Wei

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
- [x] Restricted Boltzmann Machine
- [x] Auto-encoder
- [ ] Sparse coding
- [ ] Combination of models.
- [x] Dropout
- [ ] Hessian Free Optimization
- [ ] Pre-Training

##Notes From Author:
- Neural network is really tough to learn.
- The key part in bulding a nnet is in __selecting proper parameters for layers__.
- And also, if time premitted, grabing a book about numpy is a really good choice.
- I used the `\clearPyc.py` to clean the pyc files created by python interpreter before pushing to git.

##Cautions
- It's strongly recommended to read the `doc` before using this project.
- If you have never touched __numpy__ or __theano__ before, I suggest you to try them out yourself a little bit.
It's because sometimes when you find a bug, having some knowledge of them can help great lot.

##Rules
- The input to the nnet should contain a dimension represent the _batch size_.
- The __mini-batch__ isn't supported.
However, you may see some implementations of the layers support the mini-batch, it's not supported in the optimizer.
Thought this could mean that the computation could be slower, the mini-batch can be simulated anyway.

##Acknowleges
I have learned a lot from __deeplearning.net__, part of the library is inspired by their work.
