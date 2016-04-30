#Introduction
Present the finished-models.

##Lenet model
###Introduction
This md file represents the structure of the python package lenet5.

###Modulas
####Data
The mnist datas are pre-processed. They are of size (28, 28), of hand-written digits from 0 to 9.

####Kernels
Kernels are used for __fixedGausiRbfLayer__.

####lenet
The file `myLenet.py` is a rather smaller version of __lenet5__. It can be trained really quickly and effectively.
However, the error rate of it isn't very promising.
The error rate is 3.5%, training time is 2-minutes, using 40000 instances, including the building time of the nnet.
