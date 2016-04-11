##Introduction
This modula is used to fast dump and load nnet.
Since the trainer contains several objects of class nnetController, the nnet instance is aliased in mutilple places.
Thus it will be computationally expensive to naively dump the whole trainer and it cost much more space too.
Besides of that, the load-and-dump mechanisms in this package allows you to change the dataBase dynamiclly.
Moreover, loading a previously intialized optimizer or a validator won't cost you a second time to compilte the theano.function anymore.
