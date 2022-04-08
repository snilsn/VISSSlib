import sys
import time
import glob
import os
import socket

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

import matplotlib as mpl
import numpy as np
import xarray as xr
from copy import deepcopy

import warnings
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")

import logging
import logging.config

from . import __version__
from . import time

def getMetaData(fnames, camera, config, stopAfter=-1, detectMotion4oldVersions=False, testMovieFile=True):


    nThreads = config["nThreads"]
    threshs = np.array(config["threshs"])
    goodFiles = config["goodFiles"]
    fixTime = config["fixTime"]
    
    computers = config["computers"]
    instruments = config["instruments"]

    goodFilesDict = {}
    for computer1, camera1, goodFile1 in zip(computers, instruments, goodFiles):
        goodFilesDict[camera1] = goodFile1
    goodFile = goodFilesDict[camera]

    try:
        threshs = threshs.tolist()
    except AttributeError:
        pass 

    metaDat = []
    droppedFrames = 0
    for ii in range(len(fnames)):

        metaDat1, droppedFrames1 = _getMetaData1(fnames[ii], camera, nThreads, threshs=threshs, stopAfter=stopAfter, detectMotion4oldVersions=detectMotion4oldVersions, testMovieFile=testMovieFile, goodFile = goodFile)
        droppedFrames += droppedFrames1
        if (metaDat1 is not None) and (len(metaDat1.capture_time) > 0):
            metaDat.append(metaDat1)

    if len(metaDat) == 0:
        metaDat = None
    else:
        metaDat = xr.concat(metaDat, dim='capture_time').sortby('capture_time')


    if fixTime == "fixMosaicTimeL1":
        metaDat = time.fixMosaicTimeL1(metaDat, config)

    return metaDat, droppedFrames

def _getMetaData1(fname, camera, nThreads, threshs=None, stopAfter=-1, detectMotion4oldVersions=False, testMovieFile=True,goodFile=None):

    log = logging.getLogger()

    movExtension = fname.split('.')[-1]
    
    ### meta data ####
    metaFname = fname.replace(movExtension, 'txt')

    if nThreads is None:
        record_starttime = datetime.datetime.strptime(
            fname.split('_')[-1].split('.')[0], '%Y%m%d-%H%M%S')
    else:
        record_starttime = datetime.datetime.strptime(
            fname.split('_')[-2], '%Y%m%d-%H%M%S')
        nThread = int(fname.split('_')[-1].split('.')[0])

    with open(metaFname) as f:
        firstLine = f.readline()
        if firstLine == "":
            log.error("%s: metaData empty" % fname)
            return None, 0

        if firstLine.startswith('# VISSS file format version: 0.2'):
            asciiVersion = 0.2
            asciiNames = ['capture_time', 'record_time',
                          'capture_id', 'mean', 'std']
            gitTag = f.readline().split(':')[1].lstrip().rstrip()
            gitBranch = f.readline().split(':')[1].lstrip().rstrip()
            skip = f.readline()
#        elif firstLine.startswith('# VISSS file format version: 0.3'):
#            asciiVersion = 0.3
#            asciiNames = ['capture_time', 'record_time',
#                          'capture_id'] + threshs.tolist()
#            gitTag = f.readline().split(':')[1].lstrip().rstrip()
#            gitBranch = f.readline().split(':')[1].lstrip().rstrip()
#            skip = f.readline()
        elif firstLine.startswith('# VISSS file format version: 0.3'):
            asciiVersion = 0.3
            asciiNames = ['capture_time', 'record_time',
                          'capture_id', 'queue_size'] + threshs
            gitTag = f.readline().split(':')[1].lstrip().rstrip()
            gitBranch = f.readline().split(':')[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith('# VISSS file format version: 0.4'):
            raise NotImplementedError
        elif firstLine.startswith('# VISSS file format version: 0.5'):
            raise NotImplementedError
                       
        else:
            asciiVersion = 0.1
            asciiNames = ['capture_time', 'record_time', 'capture_id']
            gitTag = '-'
            gitBranch = '-'
        capture_starttime = f.readline()
        capture_starttime = capture_starttime.split(':')[1].lstrip().rstrip()
        capture_starttime = datetime.datetime.utcfromtimestamp(
            int(capture_starttime)*1e-6)
        serialnumber = f.readline().split(':')[1].lstrip().rstrip()
        configuration = f.readline().split(':')[1].lstrip().rstrip()
        hostname = f.readline().split(':')[1].lstrip().rstrip()

    metaDat = pd.read_csv(metaFname, comment='#', names=asciiNames)
    
    #there is a frame in the video file for every line in the ASCII file (hopefully)
    metaDat.index = metaDat.index.set_names('record_id')
    metaDat = metaDat.reset_index('record_id')

    # hard to decide which variable should be used as an index:
    # - record_id: doesn't work when using multiple threads
    # - new merged record id dimension: why bother with a new variable? isn't an increasing index automatically included, ie.e. just use isel instead sel?? 
    # - capture_id: can overflow
    # - record_time: order can be mixed up after merging threads
    # - capture time: only variable that is really unique, but can be off by a couple of seconds during MOSAiC...
    metaDat = metaDat.set_index('capture_time')

    
    # time stamps are jumping around
    nJumps = np.sum(np.diff(metaDat.index) <0)
    droppedFrames = 0
    if nJumps>0:
        assert nJumps == 1, "we can handle only one jump..."
        ss = np.where(np.diff(metaDat.index) <0)[0][0]
        print(fname, 'TIME JUMPED, DROPPING %i FRAMES'%(ss+1))
        metaDat.iloc[0:ss+2] = np.nan
        droppedFrames = ss+2
        
    # very rarely, data fields are missing
    metaDat = metaDat.dropna()

    # just to be sure
    try:
        metaDat['capture_id'] = metaDat.capture_id.astype(np.int64)
    except TypeError:
        log.error("%s: metaDat[capture_id] not all int" % fname)
        return None, 0

    #fixing doesn't work with the current strategy if multiple threads are used, so deactivating
    #if metaDat.shape[0] > 1:

    #    diffs = metaDat.capture_id.diff()
    #    diffs[diffs < 0] = 1

    #    assert diffs.min() > 0

    #    newIndex = np.cumsum(diffs)
    #    newIndex[0] = 0
    #    newIndex = newIndex.astype(int)
    #else:
    #    newIndex = [0]
    #metaDat['capture_id'] = newIndex
    metaDat = xr.Dataset(metaDat)

    metaDat['capture_time'] = xr.DataArray([datetime.datetime.utcfromtimestamp(
        t1*1e-6) for t1 in metaDat['capture_time'].values], coords=metaDat['capture_time'].coords)
    metaDat['record_time'] = xr.DataArray([datetime.datetime.utcfromtimestamp(
        t1*1e-6) for t1 in metaDat['record_time'].values.astype(int)], coords=metaDat['record_time'].coords)



    metaDat['capture_starttime'] = xr.DataArray(np.array([capture_starttime]), dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['serialnumber'] = xr.DataArray([serialnumber], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['configuration'] = xr.DataArray([configuration], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['hostname'] = xr.DataArray([hostname], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['gitTag'] = xr.DataArray([gitTag], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['gitBranch'] = xr.DataArray([gitBranch], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['filename'] = xr.DataArray([fname.split('/')[-1]], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])
    metaDat['record_starttime'] = xr.DataArray([record_starttime], dims=["file_starttime"], coords=[metaDat['record_time'].values[:1]])

    if testMovieFile and (goodFile is not None) and (goodFile != "None"):
        # check for broken files:
        process = subprocess.Popen(['ffprobe', fname],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
        while True:
            output = process.stdout.readline()
            return_code = process.poll()
            if return_code is not None:
                # Process has finished, read rest of the output
                break
        if return_code == 0:
            print('OK ', fname)
            pass
        else:
            assert goodFile is not None
            print('BROKEN ', fname)
            brokenFile = '%s.broken' % fname
            repairedFile = '%s_fixed.%s' % (".".join(fname.split(".")[:-1]), 'mov')

            process = subprocess.Popen(['/home/mmaahn/bin/untrunc', goodFile, fname],
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True)

            while True:
                output = process.stdout.readline()
                print(output.strip())
                return_code = process.poll()
                if return_code is not None:
                    print('RETURN CODE', return_code)
                    # Process has finished, read rest of the output
                    for output in process.stdout.readlines():
                        print(output.strip())
                    break
            if return_code == 0:
                os.rename(fname, brokenFile)
                os.rename(repairedFile, fname)
            else:
                raise RuntimeError
            print('REPAIRED ', fname)



    if asciiVersion in [0.1, 0.2]:
        if detectMotion4oldVersions:
            log.info("%s: counting moving pixels" % (fname))

            inVid = cv2.VideoCapture(fname)

            ii = -1
            if not inVid.isOpened:
                log.error("%s: Unable to open" % fname)
                return None

            nFrames = int(inVid.get(cv2.CAP_PROP_FRAME_COUNT))

            # bug in mosaic software version, sometime there is one meta data missing
            if ((nFrames == len(metaDat.record_id) + 1) and (nFrames < 41900)):
                log.warning('%s: WARNING number of frames do not match %i %i \n' %
                                (fname, nFrames, len(metaDat.record_id)))
            elif nFrames != len(metaDat.record_id):
                log.error("%s: number of frames do not match %i %i" %
                              (fname, nFrames, len(metaDat.record_id)))
                return None

            oldFrame = None
            nChangedPixel = np.zeros(
                (len(metaDat.record_id), len(threshs)), dtype=int)

            log.info(fname)
            while True:

                ii += 1

                ret, frame = inVid.read()

                try:
                    subFrame = frame[height_offset:, :, 0]
                except TypeError:  # frame is None at the end of the file
                    break
                try:
                    nChangedPixel[ii] = checkMotion(subFrame, oldFrame, threshs)
                except IndexError:
                    log.warning(
                        '%s WARNING number of frames do not match %i %i  \n' % (fname, ii, tt))
                
                oldFrame = deepcopy(subFrame)

            inVid.release()

            metaDat['nMovingPixel'] = xr.DataArray(nChangedPixel, coords=[
                                               metaDat.record_id, xr.DataArray(
                                                   threshs, dims=['nMovingPixelThresh'])])
    # new meta data version
    else:
        metaDat['nMovingPixel'] = xr.concat([metaDat[t] for t in threshs], dim=xr.DataArray(
            threshs, dims=['nMovingPixelThresh'], name='nMovingPixelThresh')).T
        if asciiVersion in [0.3]:
            # remove threshs columns which are not needed any more due to the concat above
            metaDat = metaDat[['capture_time', 'record_time', 'capture_id', 'record_id', 'capture_starttime', 'queue_size',
                               'serialnumber', 'configuration', 'hostname', 'gitTag', 'gitBranch', 'nMovingPixel']]
            
        
        #else:
        #    metaDat = metaDat[['capture_time', 'record_time', 'capture_id', 'capture_starttime',
        #                       'serialnumber', 'configuration', 'hostname', 'gitTag', 'gitBranch', 'nMovingPixel']]

    # metaDat = metaDat.expand_dims({'record_starttime': [record_starttime]})

    metaDat['nThread'] = xr.zeros_like(metaDat['capture_id'])
    if nThreads is not None:
        metaDat['nThread'].values[:] = nThread
    else:
        metaDat['nThread'].values[:] = 0

    #save storage
    # metaDat["foundParticles"] = metaDat["foundParticles"].astype(np.uint32)
    # metaDat["movingObjects"] = metaDat["movingObjects"].astype(np.uint32)
    metaDat["nThread"] = metaDat["nThread"].astype(np.uint16)

    return metaDat, droppedFrames

def checkMotion(subFrame, oldFrame, threshs):
    '''
    Check whether something is moving - identical to C code
    '''

    if oldFrame is None:
        oldFrame = np.zeros(subFrame.shape, dtype=np.uint8)

    nChangedPixel = np.zeros(
                ( len(threshs)), dtype=int)

    absdiff = cv2.absdiff(subFrame, oldFrame)

    for tt, thresh in enumerate(threshs):
            nChangedPixel[tt] = ((absdiff >= thresh)).sum()

    return nChangedPixel
