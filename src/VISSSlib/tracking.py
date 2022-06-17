# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
#import av
import bottleneck as bn

import logging
log = logging.getLogger()

from copy import deepcopy
import warnings


from . import __version__