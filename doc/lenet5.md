##Introduction
This md file represents the structure of the python package lenet5.

##Modulas
###Data
The mnist datas are pre-processed. They are of size (28, 28), of hand-written digits from 0 to 9.

###Kernels
Kernels are used for __fixedGausiRbfLayer__.

###Lenet5
The file can be found in `\lenet5\lenet5.py`.
`\lenet5\lenet5S_track.py` is a simplified form of lenet5 and can track down the training process.

###MyLenet
The file `myLenet.py` is a rather smaller version of __lenet5__. It can be trained really quickly and effectively.
However, the error rate of it isn't very promising.
