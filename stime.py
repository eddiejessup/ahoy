from __future__ import print_function, division
from ahoy.utils.meta import make_repr_str


class Time(object):
    def __init__(self, dt):
        self.t = 0.0
        self.i = 0
        self.dt = dt

    def iterate(self):
        self.t += self.dt
        self.i += 1

    def __repr__(self):
        fs = [('t', self.t), ('i', self.i), ('dt', self.dt)]
        return make_repr_str(self, fs)
