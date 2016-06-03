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
#####Normal Lenet
The file `lenet.py` is a rather smaller version of __lenet5__. It can be trained really quickly and effectively.
However, the error rate of it isn't very promising.
The error rate is 3.5%, training time is 2-minutes, using 40000 instances, including the building time of the nnet.

#####Lenet With Dropout
With dropout, lenet can be faster to train.
The file `dropout_lenet.py` implements the dropout lenet.
__I only apply dropout for the Full-Connected Layer(The WeightLayer)__ because it's most plausible.
Also, the weights of these two layers are magnified by 2 because the annealtion-possibility of dropout is 0.5.
With using 20000 instances, the error rate is 7.5% which is slightly better than normal lenet with error rate 8.3%.
However, when the number of available instances is large, the difference isn't really great.
In fact, dropout-lenet and lenet acheive the same error rate of 3.5%.

##Restricted Boltzmann Machine
###Modulas
####rbm
This modula trains a rbm and save it. Then sampling 20 examples from it to form an image.
I used around 20000 instances to train the rbm and the results are satisfying.
Generally using around 10 to 15 epochs should be the standard but I don't have that much time.
The tiled-image can be found within this package.

####load-rbm
It's used to later examine the rbm previously dumped.
