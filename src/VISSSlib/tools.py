# -*- coding: utf-8 -*-

import yaml
import warnings
import datetime

from addict import Dict
from copy import deepcopy

import pandas as pd
import numpy as np

from . import files

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


def open_mflevel1detect(fnames, config, removeTouchBorders=True, start=None, end=None):
    '''
    helper function to open multiple level1detect files at once
    '''
    if type(fnames) is not list:
        fnames = [fnames]
        
    dats = []
    for fname in fnames:
        dat = xr.open_dataset(fname)
        ffl1 = files.FilenamesFromLevel(fname, config)

        if start is not None:
            dat = dat.isel(pid=(dat.capture_time >= start))
        if end is not None:
            dat = dat.isel(pid=(dat.capture_time <= end))
        if removeTouchBorders:
            dat = dat.isel(pid=(~dat.touchesBorder.any('side')))
            
        dat = dat.assign_coords(
        file_starttime=[ffl1.datetime64])
        dat = dat.stack(fpid=["pid", "file_starttime"])

        if len(dat.fpid) ==0:
            continue
        
        dats.append(dat)
        
    if len(dats) > 0:
        dats = xr.concat(dats, dim="fpid")
        return dats
    else:
        return None

def removeBlockedData(dat1D, events, threshold=0.1):
    '''
    remove data where window was blocked more than 10%
    '''

    blocked = (events.blocking.sel(blockingThreshold=50) > threshold)
   
    # interpolate blocking status to observed particles
    isBlocked = blocked.sel(file_starttime=dat1D.capture_time, method="nearest")
    dat1D.isel(fpid=(~isBlocked))
    
    if len(dat1D.fpid) == 0:
        print("no data after removing blocked data")
        return None
    else:
        return dat1D


def estimateCaptureIdDiff(ffl1, graceInterval=2):

    '''
    estimate capture id difference between two cameras
    '''

    mfl = xr.open_dataset(ffl1.fname.metaFrames)
    mff = xr.open_mfdataset(ffl1.filenamesOtherCamera(level="metaFrames", graceInterval=graceInterval))

    nPoints = 500
    if len(mfl.capture_time)>nPoints:
        points = np.linspace(0,len(mfl.capture_time),nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(mfl.capture_time))
    idDiffs = []
    for point in points:
        absDiff = np.abs(mfl.capture_time.isel(capture_time=point).values - mff.capture_time)
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(1,"ms"):
            pII = absDiff.argmin().values
            idDiff = mff.capture_id.values[pII] - mfl.capture_id.isel(capture_time=point).values
            idDiffs.append(idDiff)
    mfl.close()
    mff.close()
    
    if len(idDiffs)>1:
        (vals, idx, counts) = np.unique(idDiffs, return_index=True, return_counts=True)
        idDiff = vals[np.argmax(counts)]
        ratioSame = np.sum(idDiffs == idDiff)/len(idDiffs)
        if ratioSame > 0.75:
            print(f"capture_id determined {idDiff}, {ratioSame*100}% have the same value" )
            return idDiff
        else:
            print(f"capture_id varies too much, only {ratioSame*100}% have the same value {idDiff}" )
            idDiff = None

    else:
        print(f"len(idDiffs) {len(idDiffs)} is too short" )
        idDiff = None

    return idDiff