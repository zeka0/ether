__author__ = 'mac'
'''
A filter is an object designed to filter out some of the instances
Or to modify some of the properties of the instance
You should follow the design:
1st: call "should_filter(self, instance)" to determine whether you should filter it
2rd: call "filter_instance" to filter it only if you "should_filter"
'''

class filter:
    '''
    Basic filter
    Do nothing on filtering instances
    '''
    def __init__(self):
        pass

    def is_valid(self, instance):
        return True

    def should_filter(self, instance):
        return False

    def filter_instance(self, instance):
        return instance
