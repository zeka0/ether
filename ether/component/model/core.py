class model(object):
    def get_cost(self):
        raise NotImplementedError()

    def get_monitoring_cost(self):
        raise NotImplementedError()

    def get_extra_updates(self):
        raise NotImplementedError()