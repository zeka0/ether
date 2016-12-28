'''
For the purpose of debugging
This module is dedicated to implementing the naming standards of variables in the models
'''

def get_name(comp_obj, var_name):
    '''
    :param comp_obj: component object, the component that's going to hold the newly created object
    :param var_name: the name of the to-be created variable
    '''
    return comp_obj.__class__.__name__ + '.' + var_name
