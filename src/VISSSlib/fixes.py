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


def captureIdOverflows(dat, config, storeOrig=True, idOffset=0, dim="pid"):
    '''
    For M1280, capture_id is 16 bit integer and does overflow very few minutes
    fixed in raw data in version 0.3  06/2022
    '''
    maxInt = 65535

    if storeOrig:
        dat["capture_id_orig"] = deepcopy(dat["capture_id"])
    
    #constant offset
    if idOffset != 0:
        dat["capture_id"] += idOffset
    
    idDiffObserved = dat.capture_id.diff(dim)
    idDiffEstimated = np.round(
            dat.capture_time.diff(dim) / np.timedelta64(round(1/config.fps * 1e6),"us")
        ).astype(int)
    stepsObserved = (idDiffObserved<0) | (idDiffEstimated >= maxInt)
    nStepsObserved = stepsObserved.sum()

    # estimate expected steps
    firstII = dat.capture_id.values[0]
    firstCaptureT = dat.capture_time.values[0]
    lastCaptureT = dat.capture_time.values[-1]

    deltaT = (lastCaptureT-firstCaptureT)/ np.timedelta64(1, 's')
    nFrames = np.ceil(deltaT*config.fps).astype(int)
    nStepsExpected = int((firstII+nFrames)/maxInt)

    if nStepsObserved == nStepsExpected == 0:
        #nothing to do
        return dat

    if nStepsExpected == nStepsObserved:

        jumpIIs = np.where(stepsObserved)[0]+1

        for jumpII in jumpIIs:
            dat["capture_id"][jumpII:] += maxInt

    else:
        raise RuntimeError("was einfallen lassen...")

    assert np.all(dat.capture_id.diff(dim)>=0)
    print(f"captureIdOverflows: expecting {nStepsExpected} jumps, found and fixed {(stepsObserved).sum()} jumps")

    return dat

def revertIdOverflowFix(dat):
    dat = dat.rename({"capture_id": "capture_id_fixed"})
    dat = dat.rename({"capture_id_orig": "capture_id"})
    return dat

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
    '''
    check for delayed clock reset and truncate data if needed. jump needs to 
    be at least 10s to make sure we look at the right problem
    '''
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

def makeCaptureTimeEven(datF, config, dim="capture_time"):
    '''
    for the M1280 follower, the drift can be quite large so
    that clocks drifts more than 1 frame apart within 10 mins.
    Therefore, lets build a new time vector (based on a capture_id 
    that we trust) with even spacing.
    '''
    print("making follower times even")
    assert np.all(datF.capture_id.diff(dim)>=0), "capture_id mus increase monotonically "

    slopeF = ((datF["capture_time"].diff(dim) /
          datF["capture_id"].diff(dim))).astype(int)
    
    configSlope = int(round(1e9/config.fps, -3))
    deltaSlope = 1000 # =1us

    # make sure we do not have ghost frames in the data
    if dim == "pid":
        # we can have slope 0 in level1detect
        slopeF = slopeF.isel(pid=(datF["capture_id"].diff(dim) !=0))
    assert slopeF.min() >= (configSlope-deltaSlope)
    assert slopeF.max() <= (configSlope+deltaSlope)

    offset = datF.capture_time.values[0]
    fixedTime = (((datF.capture_id-datF.capture_id[0]) * configSlope)+offset)
    
    datF["capture_time_orig"] = deepcopy(datF["capture_time"])
    datF["capture_time"] = fixedTime
    
    return datF

def revertMakeCaptureTimeEven(dat):
    dat = dat.rename({"capture_time": "capture_time_even"})
    dat = dat.rename({"capture_time_orig": "capture_time"})
    return dat
