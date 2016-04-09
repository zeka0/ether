##BookNotes
Use
'''python
def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

    if hasattr(self, 'attrName'):
        pass
'''
to add the attributes into the object
