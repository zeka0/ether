from theano import tensor as T
import theano
'''
Known as supervised-cost
Support the batch cost computation

The cost function computes a mean of the costs of the mini-batch
'''
def sqrt_mean(opt):
    return T.sum( T.sqrt( opt.get_targetTensor() - opt.get_outputTensor() ) )

def cross_entro(opt):
    output = opt.get_outputTensor().dimshuffle(1, 0) #exchange dimension
    dot_product =  -T.sum(T.dot(opt.get_targetTensor(), T.log(output)), axis=1)
    return T.mean(dot_product)

def negtive_log(opt):
    def per_ins_negative_log(outputTensor, tarTensor):
        return - T.log(outputTensor)[T.argmax(tarTensor)]
    cost, updates = theano.scan(per_ins_negative_log,
                                sequences=[opt.get_outputTensor(), opt.get_targetTensor()])

    return T.mean(cost)

