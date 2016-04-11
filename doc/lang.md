##Introduction
The purpose of this Chapter is to introduce some usage of the Python programing language.
It also serves as a notebook for the author Alphasis(me).

##BookNotes
- __Use of kwargs__
 To add the attributes into the object, use
```python
def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

    if hasattr(self, 'attrName'):
        pass
```
