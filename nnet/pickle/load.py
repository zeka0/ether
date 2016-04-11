import pickle
from core import *
from nnet.util.controller import nnetController
from nnet.mlp.trainer.trainer import trainer

def load_nnet():
    try:
        with open( get_nnet_fpath(), 'rb' ) as fi:
            return pickle.load(fi)
    except Exception as ex:
        print 'Exception occured in process of loading nnet'
        print ex

def load_optimizer(nnet):
    try:
        with open( get_optimizer_fpath(), 'rb' ) as fi:
            opt = pickle.load(fi)
        nnetController.set_owner(opt, nnet)
        return opt
    except Exception as ex:
        print 'Exception occured in process of loading nnet'
        print ex

def load_validator(nnet):
    try:
        with open( get_validator_fpath(), 'rb' ) as fi:
            val = pickle.load(fi)
        nnetController.set_owner(val, nnet)
        return val
    except Exception as ex:
        print 'Exception occured in process of loading nnet'
        print ex

def load_trainer(dataBase):
    '''
    As the optimizer and the validator may have computed before,
    we skip the calling of the 'compile' in trainer.
    '''
    nnet = load_nnet()
    opt = load_optimizer(nnet)
    val = load_validator(nnet)
    tri = trainer(dataBase, opt, val, nnet)
    return tri
