import pickle

from core import *
from ether.component.model.controller import controller
from ether.debug.tracker import tracker
from ether.trainer import trainer


def load_model():
    try:
        with open( get_model_fpath(), 'rb' ) as fi:
            return pickle.load(fi)
    except Exception as ex:
        print 'Exception occured in process of loading model'
        print ex

def load_optimizer(model):
    try:
        with open( get_optimizer_fpath(), 'rb' ) as fi:
            opt = pickle.load(fi)
        controller.set_owner(opt, model)
        if isinstance(opt, tracker):
            controller.set_owner( opt.get_opt(), model )
        return opt
    except Exception as ex:
        print 'Exception occured in process of loading model'
        print ex

def load_validator(model):
    try:
        with open( get_validator_fpath(), 'rb' ) as fi:
            val = pickle.load(fi)
        controller.set_owner(val, model)
        return val
    except Exception as ex:
        print 'Exception occured in process of loading model'
        print ex

def load_trainer(dataBase):
    '''
    As the optimizer and the validator may have computed before,
    we skip the calling of the 'compile' in trainer.
    '''
    model = load_model()
    opt = load_optimizer(model)
    val = load_validator(model)
    tri = trainer(dataBase, opt, val, model)
    return tri
