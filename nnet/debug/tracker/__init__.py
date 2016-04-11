from grad import *

from layer import *

'''
In order to implement a optimizer wrapper
There are two methods you should implement
First is the set_owner method
Second is the train_once method
'''

'''
Once the set_owner method is called,
all calls for the "add" prefixed method will invalidate the whole tracker
Don't call them after set_owner is called!
(Sometimes it means don't call them once you add a tracker into a trainer)
'''

from param import *