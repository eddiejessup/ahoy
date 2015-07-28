from __future__ import print_function, division


def make_repr_str(cls, fs):
    args = ', '.join(['{}={}'.format(*f) for f in fs])
    return '{}({})'.format(cls.__class__.__name__, args)
