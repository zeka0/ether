class trainer(object):
    def compile(self):
        raise NotImplementedError()

    def train(self, cycles):
        raise NotImplementedError()

    def validate(self, cycles):
        raise NotImplementedError()