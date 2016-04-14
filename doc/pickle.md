##Introduction
This modula is used to fast dump and load nnet.
Since the trainer contains several objects of class nnetController, the nnet instance is aliased in mutilple places.
Thus it will be computationally expensive to naively dump the whole trainer and it cost much more space too.
Besides of that, the load-and-dump mechanisms in this package allows you to change the dataBase dynamiclly.
Moreover, loading a previously intialized optimizer or a validator won't cost you a second time to compilte the theano.function anymore.

##Modulas and Packages
###nnet_pickle.core
This file defines several key path for the pickle package to use
There are several things one should bear in mind.
When in dumping objects, this modula will never dump the trainer as a whole because it can be costly and un-reliable because of the database.

###dump
First thing to remember is that __in dumping, the dataBase is never dumped.__
Since the optimizer and the validator are all of the nnetController class.
We should avoid dumping the nnet within them as we will dump nnet seperately.
It can be faster and save more space.

###load
It's required to pass a __dataBase__ to the constructor of the trainer.
