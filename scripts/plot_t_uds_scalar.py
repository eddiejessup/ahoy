#!/usr/bin/env python
from __future__ import print_function, division
import sys
from ahoy.plot import plot


if __name__ == '__main__':
    plot.plot_t_uds_scalar(sys.argv[1])
