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
