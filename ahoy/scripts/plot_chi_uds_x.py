#!/usr/bin/env python
from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
from ahoy.plot import plot


if __name__ == '__main__':
    ax = plt.gca()
    plot.plot_chi_uds_x(sys.argv[1:], ax)
    plt.show()