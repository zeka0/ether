'''
In dumping, the dataBase is never dumped.
'''
import pickle

from core import *
from ether.component.model.controller import controller
from ether.debug.tracker.core import tracker


def dump_model(model):
    try:
        with open( get_model_fpath(), 'wb' ) as fi:
            pickle.dump(model, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping model'
        print ex

'''
Since the optimizer and the validator are all of the constroller class.
We should avoid dumping the model within them as we will dump model seperately.
It can be faster and save more space.
'''
def dump_optimizer(opt):
    net = opt.get_owner()
    controller.set_owner(opt, None)
    if isinstance(opt, tracker):
        controller.set_owner( opt.get_opt(), None )

    try:
        with open( get_optimizer_fpath(), 'wb' ) as fi:
            pickle.dump(opt, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping optimizer'
        print ex
    finally:
        controller.set_owner(opt, net)#preserve the validity of the optimizer
        if isinstance(opt, tracker):
            controller.set_owner(opt.get_opt(), net)#preserve the validity of the optimizer

def dump_validator(val):
    net = val.get_owner()
    controller.set_owner(val, None)
    try:
        with open( get_validator_fpath(), 'wb' ) as fi:
            pickle.dump(val, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping validator'
        print ex
    finally:
        controller.set_owner(val, net)#preserve the validity of validator

def dump_trainer(trainer):
    dump_model(trainer.get_model())
    dump_optimizer(trainer.get_optimizer())
    dump_validator(trainer.get_validator())
