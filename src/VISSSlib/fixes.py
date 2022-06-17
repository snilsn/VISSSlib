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

    quite poor attempt, not used any more!
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


def fixIntOverflow(metaDat, maxInt=65535, storeOrig=True, idOffset=0):
    '''
    For M1250, capture_id is 16 bit integer
    '''

    if storeOrig:
        metaDat["capture_id_orig"] = deepcopy(metaDat["capture_id"])

    metaDat["capture_id"] = metaDat["capture_id"] + idOffset

    overflows = np.where(metaDat.capture_id.diff("capture_time") < (-maxInt*0.01))[0]
    for overflow in overflows:

        print("overflow at position", overflow)
        metaDat["capture_id"][(overflow+1):] = metaDat.capture_id[(overflow+1):] + maxInt
    return metaDat

def removeGhostFrames(metaDat, config, intOverflow=True, idOffset = 0, fixIteration = 3):
    '''
    For MOSAiC, follower adds once in a while additional ghost frames to the data set. 
    They can be recognized because they are less than 1/fps apart. Typically, a group 
    of 6 frames has reduced 1/fps

    we do this several times because sometime multiple ghost frames are in data gaps so 
    that only the shiftes index tells that there was a problem

    if fixIteration iterations do not change it, give up and cut file at first
    suspicous position
    '''

    beyondRepair = False
    metaDat["capture_id_orig"] = deepcopy(metaDat["capture_id"])

    metaDat["capture_id"] = metaDat["capture_id"] + idOffset

    if intOverflow:
        metaDat = fixIntOverflow(metaDat, storeOrig=False, idOffset=0)

    # ns are assumed
    assert metaDat["capture_time"].dtype == '<M8[ns]'

    droppedFrames = 0
    for nn in range(fixIteration+1):
        slope = ((metaDat["capture_time"].diff("capture_time") /
                  metaDat["capture_id"].diff("capture_time"))).astype(int)
        configSlope = 1e9/config.fps
        # we find them because dat is not 1/fps apart
        jumps = ((slope/configSlope).values >
                 1.03) | ((slope/configSlope).values < 0.97)
        jumpsII = np.where(jumps)[0]
        nGroups = sum(k for k, v in groupby(jumps))

        # the last loop is only for testng 
        if nn == fixIteration:
            if nGroups != 0:
                print("FILE BROKEN BEYOND REPAIR")
                droppedFrames += len(metaDat.capture_time)-jumpsII[0]
                #remove fishy data and everything after
                metaDat = metaDat.isel(capture_time = slice(0,jumpsII[0]))
                beyondRepair = True
            break

        lastII = np.concatenate(
            (jumpsII[:-1][np.diff(jumpsII) != 1], jumpsII[-1:])) + 1
        assert nGroups == len(lastII)

        for lastI in lastII:
            metaDat["capture_id"][lastI:] = metaDat["capture_id"][lastI:]-1

        # remove all fishy frames
        metaDat = metaDat.drop_isel(capture_time=jumpsII)
        droppedFrames += len(jumpsII)

        if nGroups > 0:
            print(f"ghost iteration {nn}: found {nGroups} ghost frames at {lastII.tolist()}")
        else:
            break

    return metaDat, droppedFrames, beyondRepair

def delayedClockReset(metaDat, config):

    #check for delayed clock reset, jump needs to be at least 10s to make sure
    # we look at the right problem
    if  (metaDat.capture_time.diff() <= -10e6).any(): 
    
        resetII = np.where((metaDat.capture_time.diff() < -10e6))[0]
        assert len(resetII) == 1, "len(resetII) %i"%len(resetII)
        resetII = resetII[0] # +1 already applied by pandas!
        assert resetII < 20, "time jump usually occures within first few frames %i"%resetII

        if (metaDat.capture_id.diff()[1:resetII+1] < 0).any():
            # we cannot handle int overflows in capture id AND wrong timestamps,
            # cut data
            metaDat = metaDat.iloc[resetII:]
        else:
            #attempt to fix it!

            firstGoodTime = metaDat.capture_time.iat[resetII]
            firstGoodID = metaDat.capture_id.iat[resetII]
            deltaT = round(1/config.fps * 1e6)
            offsets = (metaDat.capture_id.iloc[:resetII] - firstGoodID ) * deltaT
            metaDat.iloc[:resetII, metaDat.columns.get_loc('capture_time')] = firstGoodTime + offsets

    return metaDat