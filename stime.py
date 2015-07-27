from __future__ import print_function, division


class Time(object):
    def __init__(self, dt):
        self.t = 0.0
        self.i = 0
        self.dt = dt

    def iterate(self):
        self.t += self.dt
        self.i += 1

    def __repr__(self):
        dct = {'t': self.t, 'i': self.i, 'dt': self.dt}
        return '{}({})' % (self.__class__, dct)
