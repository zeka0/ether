##Introduction
The purpose of this Chapter is to introduce some usage of the Python programing language.
It also serves as a notebook for the author Alphasis(me).

##BookNotes
- __Use of kwargs__ In python programing, the dynamic of the number of arguments in a method can be very crucial.
You may have programmed python for months and never used __**kwargs__ or __*args__ before.
However, in some contexts, you will find them come as very handy.
To add the attributes into the object, use
```python
def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

    if hasattr(self, 'attrName'):
        pass
```
