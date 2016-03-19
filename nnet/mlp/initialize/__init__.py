'''
Special module used to initialize parameters
Such as weights, bias, filters(conv2d) etc
One thing to remember is to create them using theano.shared function
It's required that all the shared variables of type matrix instead of vector
'''

import normal

import shared

import uniform
