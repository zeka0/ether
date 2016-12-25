from theano import tensor as T
import theano
'''
Known as supervised-cost
Support the batch cost computation
Typically return a matrix with the first dimension meaning the size of the mini-batch
'''
def sqrt_mean(opt):
    return T.sum( T.sqrt( opt.get_targetTensor() - opt.get_outputTensor() ) )

#TODO not support batch
def cross_entro(opt):
    return - ( opt.get_targetTensor().dot( ( T.log( T.flatten(opt.get_outputTensor()[0], outdim=1) / opt.get_targetTensor() ) ).T ) ).sum()

def lenet(opt):
    '''
    It follows that the output shape of a layer is a matrix regardless of whether it's vector or not
    '''
    return T.sum( opt.get_outputTensor()[:][ T.argmax(opt.get_targetTensor()) ] + T.log( 6 + ( T.exp( - opt.get_outputTensor() ) ).sum(axis=0) ) )

def negtive_log(opt):
    def per_ins_negative_log(outputTensor, tarTensor):
        return - T.log(outputTensor)[T.argmax(tarTensor)]
    cost, updates = theano.scan(per_ins_negative_log,
                                sequences=[opt.get_outputTensor(), opt.get_targetTensor()])

    return T.mean(cost)

