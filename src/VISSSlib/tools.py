# -*- coding: utf-8 -*-

import yaml
import warnings
import datetime

from addict import Dict
from copy import deepcopy

import pandas as pd
import numpy as np
import xarray as xr

from . import files
from . import fixes

LOGGING_CONFIG = { 
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': { 
        'standard': { 
        'format': "'%(asctime)s: %(levelname)s: %(name)s.%(funcName)s: %(message)s'"
        },
    },
    'handlers': { 
        'stream': { 
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # stream is stderr
        },
        'file': { 
            'level': 'WARNING',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': None,  # stream is stderr
        },    },
    'loggers': { 
        '': {  # root logger
            'handlers': ['stream', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
    } 
}

def get_logging_config(fname):
    lc = deepcopy(LOGGING_CONFIG)
    lc['handlers']['file']['filename'] = fname

    return lc

    
niceNames= (
    ('master', 'leader'),
    ('trigger', 'leader'),
    ('slave', 'follower'),
)

def nicerNames(string):
    for i in range(len(niceNames)):
        string=string.replace(*niceNames[i]) 
    return string

def readSettings(fname):
    with open(fname, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return Dict(config)

def getDateRange(nDays, config):
    if config["end"] == "today":
        end = datetime.datetime.utcnow() 
        end2 = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    else:
        end = end2 = config["end"]
    
    if nDays == 0:
        days = pd.date_range(
            start=config["start"],
            end=end2,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    elif type(nDays) is str:
        days = nDays.split(",")
        if len(days) == 1:
            days = nDays.split("-")
            if len(days) == 1:
                days = pd.date_range(
                    start=nDays,
                    periods=1,
                    freq="1D",
                    tz=None,
                    normalize=True,
                    name=None,
                    inclusive=None
                )
            else:
                days = pd.date_range(
                    start=days[0],
                    end=days[1],
                    freq="1D",
                    tz=None,
                    normalize=True,
                    name=None,
                    inclusive=None)
        else:
            days = pd.DatetimeIndex(days)

    else:
        days = pd.date_range(
            end=end2,
            periods=nDays,
            freq="1D",
            tz=None,
            normalize=True,
            name=None,
            inclusive=None
        )
    return days

def otherCamera(camera, config):
    if camera == config["instruments"][0]:
        return config["instruments"][1]
    elif camera == config["instruments"][1]:
        return config["instruments"][0]
    else:
        raise ValueError

def getMode(a):
    (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
    index = idx[np.argmax(counts)]
    mode = a[index]
    return a

def open_mfmetaFrames(fnames, config, start=None, end=None, applyFixes=True):
    '''
    helper function to open multiple metaFrame files at once
    '''

    def preprocess(dat):
        # keep track of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray([ffl1.datetime64]*len(dat.capture_time), coords=[dat.capture_time])

        return dat

    dat = xr.open_mfdataset(fnames, combine="nested", concat_dim="capture_time", preprocess=preprocess).load()

    if applyFixes:
        # fix potential integer overflows if necessary
        if "captureIdOverflows" in config.dataFixes:
            dat = fixes.captureIdOverflows(dat, config, dim="capture_time")

        # fix potential integer overflows if necessary
        if "makeCaptureTimeEven" in config.dataFixes:
            #does not make sense for leader
            if "follower" in dat.encoding["source"]:
                dat = fixes.makeCaptureTimeEven(dat, config, dim="capture_time")

    
    if start is not None:
        dat = dat.isel(capture_time=(dat.capture_time >= start))
        if len(dat.capture_time) == 0:
            return None
        
    if (end is not None):
        dat = dat.isel(capture_time=(dat.capture_time <= end))
        if len(dat.capture_time) == 0:
            return None
        
    if len(dat.capture_time) == 0:
        return None

    return dat



def open_mflevel1detect(fnames, config, removeTouchBorders=True, start=None, end=None, applyFixes=True, datVars="all"):
    '''
    helper function to open multiple level1detect files at once
    '''
    '''
    helper function to open multiple level1detect files at once
    '''

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray([ffl1.datetime64]*len(dat.pid), coords=[dat.pid])
        if "nThread" not in dat.keys():
            dat["nThread"] = xr.DataArray([-99]*len(dat.pid), coords=[dat.pid])

        if datVars != "all":
            dat = dat[datVars]

        return dat

    dat = xr.open_mfdataset(fnames, combine="nested", concat_dim="pid", preprocess=preprocess).load()

    if applyFixes:
        # fix potential integer overflows if necessary
        if "captureIdOverflows" in config.dataFixes:
            dat = fixes.captureIdOverflows(dat, config)

        # fix potential integer overflows if necessary
        if "makeCaptureTimeEven" in config.dataFixes:
            #does not make sense for leader
            if "follower" in dat.encoding["source"]:
                dat = fixes.makeCaptureTimeEven(dat, config, dim="pid")

    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pid": "fpid"})
        
    if len(dat.fpid) == 0:
        return None
    
    if start is not None:
        dat = dat.isel(fpid=(dat.capture_time >= start))
        if len(dat.fpid) == 0:
            return None
        
    if (end is not None):
        dat = dat.isel(fpid=(dat.capture_time <= end))
        if len(dat.fpid) == 0:
            return None
        
    if removeTouchBorders:
        dat = dat.isel(fpid=(~dat.touchesBorder.any('side')))

    if len(dat.fpid) == 0:
        return None

    dat.load()
    return dat

def open_mflevel1match(fnames, config):
    '''
    helper function to open multiple level1match files at once
    '''
    if type(fnames) is not list:
        fnames = [fnames]
        

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray([ffl1.datetime64]*len(dat.pair_id), coords=[dat.pair_id])
        return dat

    dat = xr.open_mfdataset(fnames, combine="nested", concat_dim="pair_id", preprocess=preprocess).load()
    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pair_id": "fpair_id"})

    return dat


def removeBlockedData(dat1D, events, threshold=0.1):
    '''
    remove data where window was blocked more than 10%
    '''
    #shortcut
    if dat1D is None:
        return None

    if type(events) is str:
        events = xr.open_dataset(events)
    blocked = (events.blocking.sel(blockingThreshold=50) > threshold)
   
    # interpolate blocking status to observed particles
    isBlocked = blocked.sel(file_starttime=dat1D.capture_time, method="nearest").values
    dat1D = dat1D.isel(fpid=(~isBlocked))
    events.close()

    if len(dat1D.fpid) == 0:
        print("no data after removing blocked data")
        return None
    else:
        return dat1D


def estimateCaptureIdDiff(leaderFile, followerFiles, config, dim, nPoints = 500, maxDiffMs = 1):

    '''
    estimate capture id difference between two cameras
    look at capture id difference of images at the "same" time
    '''
    leaderDat = xr.open_dataset(leaderFile)
    followerDat = xr.open_mfdataset(followerFiles, combine="nested", concat_dim=dim).load()

    followerDat = cutFollowerToLeader(leaderDat, followerDat, gracePeriod=0, dim=dim)

    if "captureIdOverflows" in config.dataFixes:
        leaderDat = fixes.captureIdOverflows(leaderDat, config, storeOrig=True, idOffset=0, dim=dim)
        followerDat = fixes.captureIdOverflows(followerDat, config, storeOrig=True, idOffset=0, dim=dim)
        # fix potential integer overflows if necessary

    if "makeCaptureTimeEven" in config.dataFixes:
        #does not make sense for leader
        # redo capture_time based on first time stamp...
        followerDat = fixes.makeCaptureTimeEven(followerDat, config, dim)

    idDiff =  estimateCaptureIdDiffCore(leaderDat, followerDat, dim, nPoints = nPoints, maxDiffMs = maxDiffMs)
    leaderDat.close()
    followerDat.close()
    
    return idDiff
    
def estimateCaptureIdDiffCore(leaderDat, followerDat, dim, nPoints = 500, maxDiffMs = 1, timeDim="capture_time"):
    
    if len(leaderDat[dim]) == 0:
        raise RuntimeError(f"leaderDat has zero length" )
    if len(followerDat[dim]) == 0:
        raise RuntimeError(f"followerDat has zero length" )


    if len(leaderDat[dim])>nPoints:
        points = np.linspace(0,len(leaderDat[dim]),nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(leaderDat[dim]))

    idDiffs = []
    for point in points:
        absDiff = np.abs(leaderDat[timeDim].isel(**{dim: point}).values - followerDat[timeDim])
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(int(maxDiffMs),"ms"):
            pII = absDiff.argmin().values
            idDiff = followerDat.capture_id.values[pII] - leaderDat.capture_id.isel(**{dim: point}).values
            idDiffs.append(idDiff)

    nIdDiffs = len(idDiffs)
    print(f"using {nIdDiffs} of {len(points)}")

    if nIdDiffs>1:
        (vals, idx, counts) = np.unique(idDiffs, return_index=True, return_counts=True)
        idDiff = vals[np.argmax(counts)]
        ratioSame = np.sum(idDiffs == idDiff)/nIdDiffs
        print("estimateCaptureIdDiff statistic:", dict(zip(vals[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]])))
        if ratioSame > 0.75:
            print(f"capture_id determined {idDiff}, {ratioSame*100}% have the same value" )
            return idDiff, nIdDiffs
        else:
            raise RuntimeError(f"capture_id varies too much, only {ratioSame*100}% of {len(vals)} samples have the same value {idDiff}, 2n place: {vals[np.argsort(counts)[-2]]}" )

    else:
        raise RuntimeError(f"nIdDiffs {nIdDiffs} is too short" )




def cutFollowerToLeader(leader, follower, gracePeriod=1, dim="fpid"):
    start = leader.capture_time[0].values - np.timedelta64(gracePeriod,"s")
    end = leader.capture_time[-1].values + np.timedelta64(gracePeriod,"s")
    
    if start is not None:
        follower = follower.isel({dim:(follower.capture_time >= start)})
    if end is not None:
        follower = follower.isel({dim:(follower.capture_time <= end)})

    return follower

