'''
In dumping, the dataBase is never dumped.
'''
import pickle
from core import *
from nnet.util.controller import nnetController

def dump_nnet(nnet):
    try:
        with open( get_nnet_fpath(), 'wb' ) as fi:
            pickle.dump(nnet, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping nnet'
        print ex

'''
Since the optimizer and the validator are all of the nnetController class.
We should avoid dumping the nnet within them as we will dump nnet seperately.
It can be faster and save more space.
'''
def dump_optimizer(opt):
    net = opt.get_owner()
    nnetController.set_owner(opt, None)
    try:
        with open( get_optimizer_fpath(), 'wb' ) as fi:
            pickle.dump(opt, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping optimizer'
        print ex
    finally:
        nnetController.set_owner(opt, net)#preserve the validity of the optimizer

def dump_validator(val):
    net = opt.get_owner()
    nnetController.set_owner(val, None)
    try:
        with open( get_validator_fpath(), 'wb' ) as fi:
            pickle.dump(val, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping validator'
        print ex
    finally:
        nnetController.set_owner(val, net)#preserve the validity of validator

def dump_trainer(trainer):
    dump_nnet(trainer.get_nnet())
    dump_optimizer(trainer.get_optimizer())
    dump_validator(trainer.get_validator())
