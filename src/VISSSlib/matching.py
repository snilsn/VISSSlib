# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import scipy.stats
#import av
import bottleneck as bn
import cv2

import logging
log = logging.getLogger()

from copy import deepcopy

from . import __version__