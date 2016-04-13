##Introduction

##Modulas & Classes
###class instance
Class intance is used to provide a universal interface for the _dataPool_. Its sub-class __imgInstance__ will transform the scala class label to a vector which is suitable for the _classification_validator_.

###package formate
The package formate defines several classes to read different forms of data, they all obey the interface defined by __formate.core.dataReader__.

###package pool
This package defines several modulas.
- __dataPool__ is used to store the data in a queue like structure.
- __filterPool__ is the subclass of the dataPool, and it provides filtering or modification of the data.
- __filter__ defines several filters to modify the data before store it in the dataPool. However, one of the disvantages of the dataPool because of the simplicity it provides is that __dataPool can't distinguish training and validting data, neigher can it cycle throught the whole data-set several times__. The reason is that _in order to support the on-line training as well as off-line training_. Despite these disadvantages, one can mimic them easily themselves.
