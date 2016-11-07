'''
This file defines several key path for the pickle package to use
There are several things one should bear in mind.
When in dumping objects:
This modula will never dump the trainer as a whole.
Partly because it can be costly and un-reliable because of the database.
But don't worry, the trainer can be re-assembled later on.
'''

import os

rootdir = r'E:\VirtualDesktop\nnet\dump' #specifies the root directory which dump and load work

model_fname = 'model'
optimizer_fname = 'optimizer'
validator_fname = 'validator'
pool_fname = 'pool'

def get_model_fpath():
    return os.path.join(rootdir, model_fname)

def get_optimizer_fpath():
    return os.path.join(rootdir, optimizer_fname)

def get_validator_fpath():
    return os.path.join(rootdir, validator_fname)

def get_pool_fpath():
    return os.path.join(rootdir, pool_fname)