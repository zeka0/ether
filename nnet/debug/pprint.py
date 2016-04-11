def print_net(nnet):
    '''
    Print the structure of the nnet
    '''
    layers = nnet.get_layers()
    print 'input_layer: ', layers[0]
    print 'output_shape', layers[0].get_outputShape()
    for i in xrange(1, len(layers)-1):
        print 'layer: ', layers[i]
        print 'output_shape', layers[i].get_outShape()
    print 'output_layer', layers[-1]
    print 'output_shape', layers[-1].get_outputShape()

def print_layer(layer):
    '''
    Print the structure of the layer
    '''
    print 'Layer class: ', layer
    print 'Output shape: ', layer.get_outputShape()
    params = layer.get_paras()
    if params is None:
        print 'The layer has no trainable paramters'
    else:
        print 'The layer has following parameters'
        for para in params:
            print_para(para)

def print_para(para):
    '''
    Print the information of the parameter
    Parameter must be a shared variable in theano
    '''
    print 'Name: ', para
    print 'Value: ', para.get_value()
