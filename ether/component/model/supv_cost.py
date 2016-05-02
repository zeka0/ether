from theano import tensor as T
'''
Known as supervised-cost
'''
def sqrt_mean(opt):
    return T.sqrt( opt.get_targetTensor() - opt.get_outputTensor() ).sum()

def cross_entro(opt):
    return - ( opt.get_targetTensor().dot( ( T.log( opt.get_outputTensor() / opt.get_targetTensor() ) ).T ) ).sum()

def lenet(opt):
    '''
    It follows that the output shape of a layer is a matrix regardless of whether it's vector or not
    '''
    return opt.get_outputTensor()[0][ T.argmax(opt.get_targetTensor()) ] + T.log( 6 + ( T.exp( - opt.get_outputTensor() ) ).sum() )

def negtive_log(opt):
    return -T.mean( T.log( opt.get_outputTensor() )[0][T.argmax(opt.get_targetTensor())] )
