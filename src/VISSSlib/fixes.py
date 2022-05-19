# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import groupby

import numpy as np
import xarray as xr
import pandas as pn
import bottleneck as bn

import warnings
import logging
log = logging.getLogger()


# various tools to fix bugs in the data

def fixMosaicTimeL1(dat1, config):
    '''
    attempt to fix drift of capture time with record_time
    '''

    datS = dat1[["capture_time", "record_time"]]
    datS = datS.isel(capture_time=slice(None, None, config["fps"]))
    diff = (datS.capture_time - datS.record_time)

    # no estiamte the drift
    drifts1 = []
    # group netcdf into 1 minute chunks
    index1min = diff.capture_time.resample(
        capture_time="1T", label="right").first().capture_time.values
    if len(index1min) <= 2:
        index1min = diff.capture_time.resample(
            capture_time="30s", label="right").first().capture_time.values
        if len(index1min) <= 2:
            index1min = diff.capture_time.resample(
                capture_time="10s", label="right").first().capture_time.values
            if len(index1min) <= 2:
                index1min = diff.capture_time.resample(
                    capture_time="1s", label="right").first().capture_time.values

    grps = diff.groupby_bins("capture_time", bins=index1min)

    # find max. difference in each chunk
    # this is the one were we assume it is the true dirft
    # also time stamp or max.  is needed, this is why resample cannot be used directly
    for ii, grp in grps:
        drifts1.append(grp.isel(capture_time=grp.argmax()))
    drifts = xr.concat(drifts1, dim="capture_time")

    # interpolate to original resolution
    # extrapolation required for beginning or end - works usually very good!
    driftsInt = drifts.astype(int).interp_like(dat1.capture_time, kwargs={
        "fill_value": "extrapolate"}).astype('timedelta64[ns]')

    # get best time estimate
    bestestimate = dat1.capture_time.values - driftsInt.values

#                 plt.figure()
#                 driftsInt.plot(marker="x")
#                 diff.plot()

    # replace time in nc file
    dat1["capture_time_orig"] = deepcopy(dat1["capture_time"])
    dat1 = dat1.assign_coords(capture_time=bestestimate)

    # the difference between bestestimate and capture time must jump more than 1% of the measurement interval
    timeDiff = (np.abs(((dat1.capture_time-dat1.capture_time_orig).diff(
        "capture_time")/dat1.capture_time_orig.diff("capture_time"))))
    assert np.all(timeDiff < 0.01), timeDiff.max()

    return dat1


def fixIdOverflow(metaDat, maxInt=65535):
    '''
    For MOSAiC, capture_id was 16 bit integer
    '''
    overflows = np.where(metaDat.capture_id.diff("capture_time") < (-maxInt*0.8))[0]
    for overflow in overflows:

        print("overflow at ", overflow)
        metaDat["capture_id"][(overflow+1):] = metaDat.capture_id[(overflow+1):] + maxInt
    return metaDat

def removeGhostFrames(metaDat, config, idOverflow=True):
    '''
    For MOSAiC, follower adds once in a while additional ghost frames to the data set. 
    They can be recognized because they are less than 1/fps apart. Typically, a group 
    of 6 frames has reduced 1/fps
    '''


    metaDat["capture_id_orig"] = deepcopy(metaDat["capture_id"])

    if idOverflow:
        metaDat = fixIdOverflow(metaDat)

    slope = ((metaDat["capture_time_orig"].diff("capture_time") /
              metaDat["capture_id"].diff("capture_time"))).astype(int)
    configSlope = 1e9/config.fps
    # we find them because dat is not 1/fps apart
    jumps = ((slope/configSlope).values >
             1.05) | ((slope/configSlope).values < 0.95)
    jumpsII = np.where(jumps)[0]
    nGroups = sum(k for k, v in groupby(jumps))
    print(f"found {nGroups} ghost frames")
    lastII = np.concatenate(
        (jumpsII[:-1][np.diff(jumpsII) != 1], jumpsII[-1:])) + 1
    assert nGroups == len(lastII)

    for lastI in lastII:
        metaDat["capture_id"][lastI:] = metaDat["capture_id"][lastI:]-1

    # remove all fishy frames
    metaDat = metaDat.drop_isel(capture_time=jumpsII)

    return metaDat
