#!/usr/bin/env python
from __future__ import print_function, division
import sys
import numpy as np
from ahoy.utils import utils


ud_0 = 0.17
chis, uds = utils.chi_uds_x(sys.argv[1:])
i_sort = np.argsort(chis)
chis, uds = chis[i_sort], uds[i_sort]
chi_0 = utils.curve_intersect(chis, uds, ud_0)
print('chi_0: {:g}'.format(chi_0))
