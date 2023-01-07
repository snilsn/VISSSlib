# -*- coding: utf-8 -*-

import yaml
import warnings
import datetime
import socket

from addict import Dict
from copy import deepcopy

import pandas as pd
import numpy as np
import xarray as xr
import io
import os
import tarfile
import zipfile

import IPython.display
import ipywidgets
import cv2
from PIL import Image
import skimage

from . import files
from . import fixes
from . import __version__, __versionFull__

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


#settings that stay mostly constant
DEFAULT_SETTINGS = {
    'height_offset': 64,
    'minMovingPixels': [20, 10, 5, 2, 2, 2, 2],
    'threshs': [20, 30, 40, 60, 80, 100, 120],
    'goodFiles': ['None', 'None'],
    'level1detectQuicklook': {'minBlur': 500, 'minSize': 17},
    'rotate': {},
    'maxMovingObjects': 60
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
    config = DEFAULT_SETTINGS
    with open(fname, 'r') as stream:
        config.update(yaml.load(stream, Loader=yaml.Loader))
    return Dict(config)

def getDateRange(nDays, config, endYesterday=True):

    if config["end"] == "today":
        end = datetime.datetime.utcnow() 
        if endYesterday:
            end2 = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        else:
            end2 = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    else:
        end = end2 = config["end"]
    
    if (type(nDays) is int) or type(nDays) is float:
        if nDays > 1000:
            nDays = str(nDays)
    elif (type(nDays) is str) and (len(nDays)<6):
        nDays = int(nDays)
   

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

    #double check to make sure we did not add too much
    days = days[days>=pd.Timestamp(config.start)]
    if config.end != "today":
        days = days[days<=pd.Timestamp(config.end)]
    else:
        days = days[days<=datetime.datetime.utcnow() ]

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

def open_mfmetaFrames(fnames, config, start=None, end=None, skipFixes=[]):
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

    if (skipFixes != "all"):
        # fix potential integer overflows if necessary
        if( "captureIdOverflows" in config.dataFixes) and ( "captureIdOverflows" not in skipFixes):
            print("apply fix", "captureIdOverflows")
            dat = fixes.captureIdOverflows(dat, config, dim="capture_time")

        # # beware of unintended implications down the processing chain becuase capture_time is used 
        # # in analysis module
        # # fix potential integer overflows if necessary
        # if ("makeCaptureTimeEven" in config.dataFixes) and ("makeCaptureTimeEven" not in skipFixes):
        #     print("apply fix", "makeCaptureTimeEven")
        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="capture_time")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]
    
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



def open_mflevel1detect(fnamesExt, config, start=None, end=None, skipFixes=[], datVars="all"):
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

    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]

    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    dat = xr.open_mfdataset(fnames, combine="nested", concat_dim="pid", preprocess=preprocess).load()

    if start is not None:
        dat = dat.isel(pid=(dat.capture_time >= start))
        if len(dat.pid) == 0:
            return None
        
    if (end is not None):
        dat = dat.isel(pid=(dat.capture_time <= end))
        if len(dat.pid) == 0:
            return None

    if (skipFixes != "all"):
        # fix potential integer overflows if necessary
        if( "captureIdOverflows" in config.dataFixes) and ( "captureIdOverflows" not in skipFixes):

            dat = fixes.captureIdOverflows(dat, config)

        # moved to meta data
        # # fix potential integer overflows if necessary
        # if( "makeCaptureTimeEven" in config.dataFixes) and ( "makeCaptureTimeEven" not in skipFixes):

        #     #does not make sense for leader
        #     if "follower" in dat.encoding["source"]:
        #         dat = fixes.makeCaptureTimeEven(dat, config, dim="pid")
        #     else:
        #         #make sure follower and leader data are consistent
        #         dat["capture_time_orig"] = dat["capture_time"]
    
    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pid": "fpid"})
        
    if len(dat.fpid) == 0:
        return None
        
    if len(dat.fpid) == 0:
        return None
        
    dat.load()
    return dat

def open_mflevel1match(fnamesExt, config, datVars="all"):
    '''
    helper function to open multiple level1match files at once
    '''
    if type(fnamesExt) is not list:
        fnamesExt = [fnamesExt]
        
    fnames = []
    for fname in fnamesExt:
        if fname.endswith("nodata"):
            pass
        elif fname.endswith("broken.txt"):
            pass
        elif fname.endswith("notenoughframes"):
            pass
        else:
            fnames.append(fname)
    if len(fnames) == 0:
        return None

    def preprocess(dat):
        # keep trqack of file start time
        fname = dat.encoding["source"]
        ffl1 = files.FilenamesFromLevel(fname, config)
        dat["file_starttime"] = xr.DataArray([ffl1.datetime64]*len(dat.pair_id), coords=[dat.pair_id])
        if datVars != "all":
            dat = dat[datVars]
        return dat

    dat = xr.open_mfdataset(fnames, combine="nested", concat_dim="pair_id", preprocess=preprocess).load()
    # replace pid by empty dimesnion to allow concatenating files without jumps in dimension pid
    dat = dat.swap_dims({"pair_id": "fpair_id"})

    return dat

def identifyBlowingSnowData(fnames, config, timeIndex1):
    # handle blowing snow, estimate ratio of skipped frames


    blowingSnowRatio = {}
    for cam in ["leader", "follower"]:
        # print("starting identifyBlowingSnowData", cam)
        movingObjects = []
        for fna in fnames[cam]:
            movingObjects.append(xr.open_dataset(fna).movingObjects)
        movingObjects = xr.concat(movingObjects, dim="capture_time")
        # movingObjects = xr.open_mfdataset(fnames[cam],  combine='nested', preprocess=preprocess).movingObjects.load()
        movingObjects = movingObjects.sortby("capture_time")
        tooManyMove = movingObjects > config.maxMovingObjects
        tooManyMove = tooManyMove.groupby_bins("capture_time", timeIndex1, labels=timeIndex1[:-1])
        blowingSnowRatio[cam] = tooManyMove.sum()/tooManyMove.count() #now a ratio
        # nan means nothing recorded, so no blowing snow either
        blowingSnowRatio[cam] = blowingSnowRatio[cam].fillna(0)
    blowingSnowRatio = xr.concat((blowingSnowRatio["leader"], blowingSnowRatio["follower"]), dim="camera")
    blowingSnowRatio["camera"] = ["leader", "follower"]
    blowingSnowRatio = blowingSnowRatio.rename(capture_time_bins="time")

    # print("done identifyBlowingSnowData")
    return blowingSnowRatio


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

    if np.any(isBlocked) and (len(dat1D.fpid) > 0):
        print(f"{np.sum(isBlocked)/ len(dat1D.fpid)*100}% blocked data removed")

    if len(dat1D.fpid) == 0:
        print("no data after removing blocked data")
        return None
    else:
        return dat1D


def estimateCaptureIdDiff(leaderFile, followerFiles, config, dim, concat_dim="capture_time", nPoints = 500, maxDiffMs = 1):

    '''
    estimate capture id difference between two cameras
    look at capture id difference of images at the "same" time
    '''
    leaderDat = xr.open_dataset(leaderFile)
    followerDat = xr.open_mfdataset(followerFiles, combine="nested", concat_dim=concat_dim).load()

    followerDat = cutFollowerToLeader(leaderDat, followerDat, gracePeriod=0, dim=concat_dim)

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

    #check whether correction has been applied and use if present
    if (timeDim == "capture_time") and ("capture_time_even" in followerDat.data_vars):
        timeDimFollower = "capture_time_even"
    else:
        timeDimFollower = timeDim

    # cut number of investigated points in time if required
    if len(leaderDat[dim])>nPoints:
        points = np.linspace(0,len(leaderDat[dim]),nPoints, dtype=int, endpoint=False)
    else:
        points = range(len(leaderDat[dim]))

    # loop through all points
    idDiffs = []
    for point in points:
        absDiff = np.abs(leaderDat[timeDim].isel(**{dim: point}).values - followerDat[timeDimFollower])
        pMin = np.min(absDiff).values
        if pMin < np.timedelta64(int(maxDiffMs),"ms"):
            pII = absDiff.argmin().values
            idDiff = followerDat.capture_id.values[pII] - leaderDat.capture_id.isel(**{dim: point}).values
            idDiffs.append(idDiff)

    nIdDiffs = len(idDiffs)
    print(f"using {nIdDiffs} of {len(points)}")

    if nIdDiffs>0:
        (vals, idx, counts) = np.unique(idDiffs, return_index=True, return_counts=True)
        idDiff = vals[np.argmax(counts)]
        ratioSame = np.sum(idDiffs == idDiff)/nIdDiffs
        print("estimateCaptureIdDiff statistic:", dict(zip(vals[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]])), timeDim)
        if ratioSame > 0.7:
            print(f"capture_id determined {idDiff}, {ratioSame*100}% have the same value" )
            return idDiff, nIdDiffs
        else:
            raise RuntimeError(f"capture_id varies too much, only {ratioSame*100}% of {len(vals)} samples have the same value {idDiff}, 2n place: {vals[np.argsort(counts)[-2]]}" )

    else:
        print(f"nIdDiffs {nIdDiffs} is too short" )
        return None, nIdDiffs


def getOtherCamera(config, camera):
    if camera == config.instruments[0]:
        return config.instruments[1]
    elif camera == config.instruments[1]:
        return config.instruments[0]
    else:
        raise ValueError


def cutFollowerToLeader(leader, follower, gracePeriod=1, dim="fpid"):
    start = leader.capture_time[0].values - np.timedelta64(gracePeriod,"s")
    end = leader.capture_time[-1].values + np.timedelta64(gracePeriod,"s")
    
    if start is not None:
        follower = follower.isel({dim:(follower.capture_time >= start)})
    if end is not None:
        follower = follower.isel({dim:(follower.capture_time <= end)})

    return follower

def nextCase(case):
    return str(np.datetime64(f"{case[:4]}-{case[4:6]}-{case[6:8]}") + np.timedelta64(1,"D")).replace("-","")

def prevCase(case):
    return str(np.datetime64(f"{case[:4]}-{case[4:6]}-{case[6:8]}") - np.timedelta64(1,"D")).replace("-","")

def displayImage(frame, doDisplay=True, rescale = None):

    #opencv cannot handle grayscale png with alpha channel
    if (len(frame.shape) == 3) and (frame.shape[2] == 2):
        fill_color = 0
        frameAlpha = frame[:,:,1]
        frameData = frame[:,:,0]
        frameCropped = deepcopy(frameData)
        frameCropped[frameAlpha==0] = fill_color
        frame = np.hstack((frameData, frameCropped ) )

    if rescale is not None:
        frame = skimage.transform.resize(frame,
                               np.array(frame.shape)*rescale,
                               mode='edge',
                               anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               preserve_range=True,
                               order=0)

    _, frame = cv2.imencode('.png', frame)

    if doDisplay:
        IPython.display.display(IPython.display.Image(data=frame.tobytes()))
    else:
        return IPython.display.Image(data=frame.tobytes())

'''
monkey patch standard tarfile.TarFile class extended with a special function to add and read a png file

PIL instead of open cv is used because the latter does not support grayscale images with alpha channel 
'''
def _addimage(self, fname, img):

    assert fname.endswith("png")

    # encode
    img = Image.fromarray(img)
    buf1 = io.BytesIO()
    img.save(buf1, format='PNG', compress_level=9)

    # convert to uint8
    buf2 = np.frombuffer(buf1.getbuffer(), dtype=np.uint8)

    #io buf
    io_buf = io.BytesIO(buf2)

    #file info
    info = tarfile.TarInfo(name=fname)
    info.size = buf2.size

    #add file
    self.addfile(info, io_buf)        

def _extractimage(self, fname):
    handle = self.extractfile(fname)
    image = handle.read()
    image = np.array(Image.open(io.BytesIO(image)))
    return image

imageTarFile = tarfile.TarFile
imageTarFile.addimage = _addimage
imageTarFile.extractimage = _extractimage

import zipfile
class imageZipFile(zipfile.ZipFile):
    def addimage(self, fname, img):
        # encode
        img = Image.fromarray(img)
        buf1 = io.BytesIO()
        img.save(buf1, format='PNG', compress_level=9)
        # convert to uint8
        buf2 = np.frombuffer(buf1.getbuffer(), dtype=np.uint8)

        #add file
        return self.writestr(fname, buf2)        

    def addnpy(self, fname, array):
        # encode
        buf1 = io.BytesIO()
        np.save(buf1, array)
        return self.writestr(fname, buf1.getbuffer()) 
    
    def extractimage(self, fname):
        image = self.read(fname)
        image = np.array(Image.open(io.BytesIO(image)))
        return image

    def extractnpy(self, fname):
        array = np.load(io.BytesIO(self.read(fname)))
        return array


def ncAttrs(extra={}):
    attrs = {
        "VISSSlib-version": __versionFull__,
        "OpenCV-version": cv2.__version__,
        "host": socket.getfqdn(),
        "creation-time": str(datetime.datetime.utcnow()),
        }
    if  os.environ.get('USER') is not None:
        attrs["user"] =  os.environ.get('USER')

    attrs.update(extra)
    return attrs

def finishNc(dat, extra={} ):

    dat.attrs.update(ncAttrs(extra=extra))

    for k in list(dat.data_vars) + list(dat.coords):

        if dat[k].dtype == np.float64:
            dat[k] = dat[k].astype(np.float32)

        dat[k].encoding = {}
        dat[k].encoding["zlib"] = True
        dat[k].encoding["complevel"] = 5
        #need to overwrite units becuase leting xarray handle that might lead to inconsistiencies
        #due to mixing of milli and micro seconds
        if k.endswith("time") or k.endswith("time_orig") or k.endswith("time_even"):
            dat[k].encoding["units"] = 'microseconds since 1970-01-01 00:00:00'
    return dat

def getPrevRotationEstimate(datetime64, key, config):

    rotate_all = {np.datetime64(datetime.datetime.strptime(d.ljust(15, "0"), "%Y%m%d-%H%M%S")):r[key] for d,r in config.rotate.items()}
    rotTimes = np.array(list(rotate_all.keys()))
    rotDiff = datetime64 -rotTimes 
    rotTimes = rotTimes[rotDiff>np.timedelta64(0)]
    rotDiff = rotDiff[rotDiff>np.timedelta64(0)]
    
    try:
        prevTime = rotTimes[np.argmin(rotDiff)]
    except ValueError:
        raise RuntimeError(f"datetime64 {datetime64} before earliest rotation estimate {np.min(np.array(list(rotate_all.keys())))}")
    return rotate_all[prevTime]
