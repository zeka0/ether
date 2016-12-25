'''
Extra-operation base file
The intuition to create the extra-operation is that no-matter how delicate the structure is
We still have some operations needs to be done
So here it is, the extra-operation modula
It's dedicated to solve following dilemas
Ep stands for extra operation
1. Ep during a single cycle of traning process
2. Ep during a single cycle of validating process
'''

class EopBase:
    def call(self):
        raise NotImplementedError()

class DropoutResetEop(EopBase):
    '''
    Used to specially update the dropout layer's bitvec
    '''
    def __init__(self, *dplayers):
        self.dplayers = dplayers

    def call(self):
        for dpl in self.dplayers:
            dpl.reset_bitvec()