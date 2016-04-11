'''
This file defines several key path for the pickle package to use
There are several things one should bear in mind.
When in dumping objects:
This modula will never dump the trainer as a whole.
Partly because it can be costly and un-reliable because of the database.
But don't worry, the trainer can be re-assembled later on.
'''

import os

rootdir = r'.' #specifies the root directory which dump and load work

nnet_fname = 'nnet'
optimizer_fname = 'optimizer'
validator_fname = 'validator'

def get_nnet_fpath():
    return os.path.join(rootdir, nnet_fname)

def get_optimizer_fpath():
    return os.path.join(rootdir, optimizer_fname)

def get_validator_fpath():
    return os.path.join(rootdir, validator_fname)
