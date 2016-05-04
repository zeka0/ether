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

##Restricted Boltzmann Machine
###Modulas
####rbm
This modula trains a rbm and save it. Then sampling 20 examples from it to form an image.
I used around 20000 instances to train the rbm and the results are satisfying.
Generally using around 10 to 15 epochs should be the standard but I don't have that much time.
The tiled-image can be found within this package.

####load-rbm
It's used to later examine the rbm previously dumped.
