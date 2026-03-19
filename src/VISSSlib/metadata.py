import datetime
import glob
import logging
import os
import socket
import subprocess
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr
from loguru import logger as log

from . import __version__, detection, files, fixes, quicklooks, tools

warnings.filterwarnings("ignore", category=UserWarning)


"""
Mosaic problems with metadata:

* capture_id overflows at 65535 - relatively easy to fix
* capture_time drifts becuase it is only reset when camera starts
* record_time is assigned in the processing queue, can be a couple of seconds 
  off if queue is long. more importantly, it appears to drift sometimes, a stable 
  capture_id offset cannot be always obtained... 
* flipped capture_time: once in a while timestamps of two consecutive frames are 
  flipped. Not clear whetehr frame itself is also flipped. So far only 
  observed for follower. solution: remove flipped frames.
* invented frames: sometimes a couple of frames are less than 1/fps apart. 
  looks like an additonal frame is inserted and delta t is reduced accordingly
  for approx. 6 frames to make up for additional frame. Origin of additonal 
  frame is unclear
* file_starttime: obtained from record_time, so problems with record_time apply apply

"""


@log.catch(reraise=True)
def getMetaData(
    fnames,
    camera,
    config,
    stopAfter=-1,
    testMovieFile=False,
    includeHeader=False,
    idOffset=0,
    fixIteration=3,
):
    """
    Get metadata from video files for a given camera.

    This function processes metadata files from video recordings and extracts
    timing and capture information for each frame. It handles various issues
    such as timestamp jumps, ghost frames, and capture_id overflows.

    Parameters
    ----------
    fnames : str or list
        Filename or list of filenames for metadata files
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary containing processing parameters
    stopAfter : int, optional
        Stop processing after specified number of frames, default is -1 (all)
    testMovieFile : bool, optional
        Whether to test movie files for integrity, default is False
    includeHeader : bool, optional
        Whether to include header information, default is False
    idOffset : int, optional
        Offset for capture IDs, default is 0
    fixIteration : int, optional
        Iteration count for fixing issues, default is 3

    Returns
    -------
    tuple
        Tuple containing (metaDat, droppedFrames, beyondRepair) where:
        - metaDat: xarray Dataset with metadata or None if no data
        - droppedFrames: number of frames dropped due to issues
        - beyondRepair: boolean indicating if data is beyond repair
    """
    if type(fnames) is str:
        fnames = [fnames]

    nThreads = config["nThreads"]
    threshs = np.array(config.level1detect.threshs)
    goodFiles = config["goodFiles"]

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

    beyondRepair = False
    metaDat = []
    for ii in range(len(fnames)):
        metaDat1 = _getMetaData1(
            fnames[ii],
            camera,
            config,
            stopAfter=stopAfter,
            testMovieFile=testMovieFile,
            goodFile=goodFile,
            includeHeader=includeHeader,
        )
        if (metaDat1 is not None) and (len(metaDat1.capture_time) > 0):
            metaDat.append(metaDat1)

    if len(metaDat) == 0:
        metaDat = None
        droppedFrames = 0
    else:
        metaDat = xr.concat(metaDat, dim="capture_time").sortby("capture_time")

        droppedFrames = 0

        # fix time stamps are jumping around
        jumps = np.diff(metaDat.capture_time.astype(int)) < 0
        nJumps = np.sum(jumps)
        droppedIndices = []
        if nJumps > 0:
            # At mosaic, it looks like frames are sometimes swapped when caption_id overflows
            # dont try to fix but remove meta data
            if config.visssGen == "visss":
                ss = np.where(jumps)[0]
                assert nJumps < 20, "more than 20 is very fishy..."
                if len(ss) > 1:
                    assert np.all(np.diff(ss) == 1), (
                        "if there is more than one "
                        "time index going backwards, they must be in a group"
                    )
                for s1 in ss:
                    print(fnames, "TIME JUMPED, DROPPING FRAMES around %i " % (s1))
                    droppedIndices.append(s1 - 1)
                    droppedIndices.append(s1)
                    droppedIndices.append(s1 + 1)
                droppedIndices = np.unique(droppedIndices)
                metaDat = metaDat.drop_isel(capture_time=droppedIndices)
            elif config.visssGen == "visss2":
                if nJumps == 1:  # teh usual at the beginnign of the file
                    raise RuntimeError("develop fix!!!")
                    ss = np.where(jumps)[0][0]
                    droppedIndices = list(range(ss + 1))
                    metaDat = metaDat.drop_isel(capture_time=droppedIndices)
                else:
                    raise NotImplementedError
            else:
                raise RuntimeError("unknown VISSS generation %s" % config.visssGen)

        droppedFrames += len(droppedIndices)
        # end fix time stamps are jumping around

        # unclear whether it works, MX oct 2023
        # if "removeGhostFrames" in config.dataFixes:
        #     #### fix capture_id ####
        #     metaDat, droppedFrames1, beyondRepair = fixes.removeGhostFrames(metaDat, config, intOverflow=True, idOffset=idOffset, fixIteration=fixIteration)
        #     droppedFrames += droppedFrames1
        #### end fix capture_id ####
        # elif config.model == "M1280": # does not really work with all the data gaps...
        #     metaDat = fixes.fixIntOverflow(metaDat,idOffset=idOffset)

        # if "makeCaptureTimeEven" in config.fixes:
        #     if camera == config.follower:
        #         metaDat = makeCaptureTimeEven(metaDat, config)

    return metaDat, droppedFrames, beyondRepair


def _readHeaderData(fname, returnLasttime=False):
    """
    Read header information from metadata file.

    This function parses the header section of a metadata file to extract
    configuration and timing information.

    Parameters
    ----------
    fname : str
        Path to the metadata file
    returnLasttime : bool, optional
        Whether to return last timestamp information, default is False

    Returns
    -------
    tuple
        Tuple containing header information:
        - record_starttime: datetime of record start
        - asciiVersion: ASCII file version
        - gitTag: Git tag
        - gitBranch: Git branch
        - capture_starttime: datetime of capture start
        - capture_firsttime: datetime of first capture
        - capture_lasttime: datetime of last capture (if returnLasttime=True)
        - last_id: last capture ID (if returnLasttime=True)
        - serialnumber: camera serial number
        - configuration: camera configuration
        - hostname: host machine name
        - cameraTemperature: camera temperature
        - transferQueueCurrentBlockCount: current block count in transfer queue
        - transferMaxBlockSize: maximum block size in transfer queue
        - ptpStatus: PTP status
    """
    log = logging.getLogger()

    try:
        record_starttime = datetime.datetime.strptime(
            fname.split("_")[-1].split(".")[0], "%Y%m%d-%H%M%S"
        )
    except ValueError:
        record_starttime = datetime.datetime.strptime(
            fname.split("_")[-2], "%Y%m%d-%H%M%S"
        )

    with open(fname) as f:
        firstLine = f.readline()
        if firstLine == "":
            log.error("%s: metaData empty" % fname)
            return [record_starttime] + [None] * 14

        if firstLine.startswith("# VISSS file format version: 0.2"):
            asciiVersion = 0.2
            gitTag = f.readline().split(":")[1].lstrip().rstrip()
            gitBranch = f.readline().split(":")[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith("# VISSS file format version: 0.3"):
            asciiVersion = 0.3
            gitTag = f.readline().split(":")[1].lstrip().rstrip()
            gitBranch = f.readline().split(":")[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith("# VISSS file format version: 0.4"):
            asciiVersion = 0.4
            gitTag = f.readline().split(":")[1].lstrip().rstrip()
            gitBranch = f.readline().split(":")[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith("# VISSS file format version: 0.5"):
            asciiVersion = 0.5
            gitTag = f.readline().split(":")[1].lstrip().rstrip()
            gitBranch = f.readline().split(":")[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith("# VISSS file format version: 0.6"):
            asciiVersion = 0.6
            gitTag = f.readline().split(":")[1].lstrip().rstrip()
            gitBranch = f.readline().split(":")[1].lstrip().rstrip()
            skip = f.readline()
        elif firstLine.startswith("# VISSS file format version: 0.7"):
            raise NotImplementedError

        else:
            asciiVersion = 0.1
            gitTag = "-"
            gitBranch = "-"
        capture_starttime = f.readline()
        capture_starttime = capture_starttime.split(":")[1].lstrip().rstrip()
        capture_starttime = datetime.datetime.fromtimestamp(
            int(capture_starttime) * 1e-6, datetime.UTC
        )

        serialnumber = f.readline().split(":")[1].lstrip().rstrip()
        configuration = f.readline().split(":")[1].lstrip().rstrip()
        hostname = f.readline().split(":")[1].lstrip().rstrip()

        if asciiVersion >= 0.5:
            cameraTemperature = float(f.readline().split(":")[1].lstrip().rstrip())
            transferQueueCurrentBlockCount = int(
                f.readline().split(":")[1].lstrip().rstrip()
            )
            transferMaxBlockSize = float(f.readline().split(":")[1].lstrip().rstrip())

        else:
            cameraTemperature = -99.0
            transferQueueCurrentBlockCount = -99
            transferMaxBlockSize = -99.0

        if asciiVersion >= 0.6:
            ptpStatus = f.readline().split(":")[1].lstrip().rstrip()
        else:
            ptpStatus = "Disabled"

        _ = f.readline()
        capture_firsttime = f.readline()
        capture_firsttime = capture_firsttime.split(",")[0].lstrip().rstrip()
        try:
            capture_firsttime = datetime.datetime.fromtimestamp(
                int(capture_firsttime) * 1e-6, datetime.UTC
            )

        except ValueError:
            capture_firsttime = None

    if returnLasttime:
        lastLines = os.popen("tail -n 2 %s" % fname)
        secondButLastLine = lastLines.readline()
        lastLine = lastLines.readline()
        lastLines.close()

        if lastLine.startswith("# Capture time"):
            # ASCII file contains no data!
            capture_lasttime = capture_firsttime
            last_id = None
        elif lastLine.startswith("# Last capture time"):
            # meta data version 0.4 and later
            capture_lasttime = int(lastLine.split(":")[1])
            capture_lasttime = datetime.datetime.fromtimestamp(
                int(capture_lasttime) * 1e-6, datetime.UTC
            )
            last_id = None
        else:
            try:
                capture_lasttime = lastLine.split(",")[0].lstrip().rstrip()
                capture_lasttime = datetime.datetime.fromtimestamp(
                    int(capture_lasttime) * 1e-6, datetime.UTC
                )
                last_id = lastLine.split(",")[2].lstrip().rstrip()
            except (IndexError, ValueError):
                print("last line incomplete, using second but last line", fname)
                capture_lasttime = secondButLastLine.split(",")[0].lstrip().rstrip()
                capture_lasttime = datetime.datetime.fromtimestamp(
                    int(capture_lasttime) * 1e-6, datetime.UTC
                )
                last_id = secondButLastLine.split(",")[2].lstrip().rstrip()
    else:
        capture_lasttime = None
        last_id = None
    return (
        record_starttime,
        asciiVersion,
        gitTag,
        gitBranch,
        capture_starttime,
        capture_firsttime,
        capture_lasttime,
        last_id,
        serialnumber,
        configuration,
        hostname,
        cameraTemperature,
        transferQueueCurrentBlockCount,
        transferMaxBlockSize,
        ptpStatus,
    )


def _getMetaData1(
    metaFname,
    camera,
    config,
    stopAfter=-1,
    testMovieFile=True,
    goodFile=None,
    includeHeader=True,
    version=__version__,
):
    """
    Get metadata from a single metadata file.

    This internal function processes a single metadata file and extracts
    frame-level information including timing, capture IDs, and motion statistics.

    Parameters
    ----------
    metaFname : str
        Path to the metadata file
    camera : str
        Camera identifier
    config : dict
        Configuration dictionary containing processing parameters
    stopAfter : int, optional
        Stop processing after specified number of frames, default is -1 (all)
    testMovieFile : bool, optional
        Whether to test movie files for integrity, default is True
    goodFile : str, optional
        Good file reference for repair, default is None
    includeHeader : bool, optional
        Whether to include header information, default is True
    version : str, optional
        Version string, default is __version__

    Returns
    -------
    xarray.Dataset or None
        Dataset with metadata or None if no valid data
    """
    import cv2
    import pandas as pd

    nThreads = config["nThreads"]
    threshs = np.array(config.level1detect.threshs)

    log = logging.getLogger()

    ### meta data ####
    fname = metaFname.replace("txt", config.movieExtension)

    if nThreads is None:
        nThread = 0
    else:
        nThread = int(fname.split("_")[-1].split(".")[0])

    res = _readHeaderData(metaFname, returnLasttime=False)
    (
        record_starttime,
        asciiVersion,
        gitTag,
        gitBranch,
        capture_starttime,
        capture_firsttime,
        capture_lasttime,
        last_id,
        serialnumber,
        configuration,
        hostname,
        cameraTemperature,
        transferQueueCurrentBlockCount,
        transferMaxBlockSize,
        ptpStatus,
    ) = res

    if record_starttime is None:
        return None

    if asciiVersion == 0.1:
        asciiNames = ["capture_time", "record_time", "capture_id"]
    elif asciiVersion == 0.2:
        asciiNames = [
            "capture_time",
            "record_time",
            "capture_id",
            "mean",
            "std",
        ]
    elif (
        (asciiVersion == 0.3)
        and (config.site == "lim")
        and (record_starttime < datetime.datetime(2021, 7, 1))
    ):  # intermediate version, 2021-07-01  is a rough estinate
        asciiNames = [
            "capture_time",
            "record_time",
            "capture_id",
        ] + list(threshs)
        asciiVersion = "0.3a"
    elif asciiVersion in [0.3, 0.4, 0.5, 0.6]:
        asciiNames = ["capture_time", "record_time", "capture_id", "queue_size"] + list(
            threshs
        )
    else:
        raise ValueError(f"unknown asciiVersion {asciiVersion}")

    metaDat = pd.read_csv(metaFname, comment="#", names=asciiNames)

    # there is a frame in the video file for every line in the ASCII file (hopefully)
    metaDat.index = metaDat.index.set_names("record_id")
    metaDat = metaDat.reset_index("record_id")

    # check for delayed clock reset, jump needs to be at least 10s to make sure
    # we look at the right problem
    metaDat = fixes.delayedClockReset(metaDat, config)

    # hard to decide which variable should be used as an index:
    # - record_id: doesn't work when using multiple threads
    # - new merged record id dimension: why bother with a new variable? isn't an increasing index automatically included, ie.e. just use isel instead sel??
    # - capture_id: can overflow
    # - record_time: order can be mixed up after merging threads
    # - capture time: only variable that is really unique, but can be off by a couple of seconds during MOSAiC...
    metaDat = metaDat.set_index("capture_time")
    # very rarely, data fields are missing
    metaDat = metaDat.dropna()

    # fixing doesn't work with the current strategy if multiple threads are used, so deactivating
    # if metaDat.shape[0] > 1:

    #    diffs = metaDat.capture_id.diff()
    #    diffs[diffs < 0] = 1

    #    assert diffs.min() > 0

    #    newIndex = np.cumsum(diffs)
    #    newIndex[0] = 0
    #    newIndex = newIndex.astype(int)
    # else:
    #    newIndex = [0]
    # metaDat['capture_id'] = newIndex
    metaDat = xr.Dataset(metaDat)
    # just to be sure
    try:
        metaDat["capture_id"] = metaDat.capture_id.astype(np.int64)
    except:
        # typically the last one is broken
        metaDatTmp = metaDat.drop_isel(capture_time=-1)
        # try again
        try:
            metaDatTmp["capture_id"] = metaDatTmp.capture_id.astype(np.int64)
            metaDat = metaDatTmp
        except:
            log.error("%s: metaDat[capture_id] not all int" % fname)
            return None

    metaDat["capture_time"] = xr.DataArray(
        pd.to_datetime(
            [
                datetime.datetime.utcfromtimestamp(t1 * 1e-6)
                for t1 in metaDat["capture_time"].values
            ]
        ),
        coords=metaDat["capture_time"].coords,
    )
    metaDat["record_time"] = xr.DataArray(
        pd.to_datetime(
            [
                datetime.datetime.utcfromtimestamp(t1 * 1e-6)
                for t1 in metaDat["record_time"].values.astype(int)
            ]
        ),
        coords=metaDat["record_time"].coords,
    )

    if includeHeader:
        metaDat["capture_starttime"] = xr.DataArray(
            np.array([capture_starttime]),
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["serialnumber"] = xr.DataArray(
            [serialnumber],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["configuration"] = xr.DataArray(
            [configuration],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["hostname"] = xr.DataArray(
            [hostname],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["gitTag"] = xr.DataArray(
            [gitTag],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["gitBranch"] = xr.DataArray(
            [gitBranch],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["filename"] = xr.DataArray(
            [fname.split("/")[-1]],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )
        metaDat["record_starttime"] = xr.DataArray(
            [record_starttime],
            dims=["file_starttime"],
            coords=[metaDat["record_time"].values[:1]],
        )

    if testMovieFile and (goodFile is not None) and (goodFile != "None"):
        # check for broken files:
        process = subprocess.Popen(
            ["ffprobe", fname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        while True:
            output = process.stdout.readline()
            return_code = process.poll()
            if return_code is not None:
                # Process has finished, read rest of the output
                break
        if return_code == 0:
            print("OK ", fname)
            pass
        else:
            assert goodFile is not None
            print("BROKEN ", fname)
            brokenFile = "%s.broken" % fname
            repairedFile = "%s_fixed.%s" % (".".join(fname.split(".")[:-1]), "mov")

            process = subprocess.Popen(
                ["/home/mmaahn/bin/untrunc", goodFile, fname],
                stdout=subprocess.PIPE,
                universal_newlines=True,
            )

            while True:
                output = process.stdout.readline()
                print(output.strip())
                return_code = process.poll()
                if return_code is not None:
                    print("RETURN CODE", return_code)
                    # Process has finished, read rest of the output
                    for output in process.stdout.readlines():
                        print(output.strip())
                    break
            if return_code == 0:
                os.rename(fname, brokenFile)
                os.rename(repairedFile, fname)
            else:
                raise RuntimeError
            print("REPAIRED ", fname)

    if asciiVersion in [0.1, 0.2]:
        fn = files.Filenames(metaFname, config, version=version)
        helperFname = f'{config["pathOut"].format(level="metaFrames_nMovingPixel", version=version)}/{config.site}_{camera}_{fn.year}{fn.month}{fn.day}.nc'
        if os.path.isfile(helperFname):
            # for MOSAiC we can use an exisiting estimate of the numbe rof moving pixels even though it is not in the ASCII data
            helperDat = xr.open_dataset(helperFname)
            try:
                helperDat = helperDat.sel(record_starttime=fn.datetime64, drop=True)
                helperDat = helperDat.sel(record_id=metaDat.record_id.values, drop=True)
                nMovingPixels = helperDat["nMovingPixel"].values
            except KeyError:
                print("DID not find nMovingPixels")
                nMovingPixels = (
                    np.zeros(
                        (len(metaDat.capture_time), len(helperDat.nMovingPixelThresh))
                    )
                    - 9999
                )

            metaDat["nMovingPixel"] = xr.DataArray(
                nMovingPixels,
                coords=[metaDat.capture_time, helperDat.nMovingPixelThresh],
            )
            helperDat.close()
        else:
            log.warning(
                "%s: helper file not found, counting moving pixels" % (helperFname)
            )

            inVid = cv2.VideoCapture(fname)

            ii = -1
            if not inVid.isOpened:
                log.error("%s: Unable to open" % fname)
                return None

            nFrames = int(inVid.get(cv2.CAP_PROP_FRAME_COUNT))

            # bug in mosaic software version, sometime there is one meta data missing
            if (nFrames == len(metaDat.record_id) + 1) and (nFrames < 41900):
                log.warning(
                    "%s: WARNING number of frames do not match %i %i \n"
                    % (fname, nFrames, len(metaDat.record_id))
                )
            elif nFrames != len(metaDat.record_id):
                log.error(
                    "%s: number of frames do not match %i %i"
                    % (fname, nFrames, len(metaDat.record_id))
                )
                return None

            oldFrame = None
            nChangedPixel = np.zeros((len(metaDat.record_id), len(threshs)), dtype=int)

            log.info(fname)
            while True:
                ii += 1

                ret, frame = inVid.read()

                try:
                    subFrame = frame[config.level1detect.height_offset :, :, 0]
                except TypeError:  # frame is None at the end of the file
                    break
                try:
                    nChangedPixel[ii] = detection.checkMotion(
                        subFrame, oldFrame, threshs
                    )

                except IndexError:
                    log.warning(
                        "%s WARNING number of frames do not match %i  \n" % (fname, ii)
                    )

                oldFrame = deepcopy(subFrame)

            inVid.release()

            metaDat = metaDat.assign_coords({"nMovingPixelThresh": threshs})
            metaDat["nMovingPixel"] = xr.DataArray(
                nChangedPixel,
                dims=[
                    "capture_time",
                    "nMovingPixelThresh",
                ],
            )
    # new meta data version
    else:
        metaDat["nMovingPixel"] = xr.concat(
            [metaDat[t] for t in threshs],
            dim=xr.DataArray(
                threshs, dims=["nMovingPixelThresh"], name="nMovingPixelThresh"
            ),
        ).T
        if asciiVersion in [0.3, 0.4, 0.5, 0.6]:
            # remove threshs columns which are not needed any more due to the concat above
            if includeHeader:
                metaDat = metaDat[
                    [
                        "capture_time",
                        "record_time",
                        "capture_id",
                        "record_id",
                        "capture_starttime",
                        "queue_size",
                        "serialnumber",
                        "configuration",
                        "hostname",
                        "gitTag",
                        "gitBranch",
                        "nMovingPixel",
                    ]
                ]
            else:
                metaDat = metaDat[
                    [
                        "capture_time",
                        "record_time",
                        "capture_id",
                        "record_id",
                        "queue_size",
                        "nMovingPixel",
                    ]
                ]
        elif asciiVersion in ["0.3a"]:
            if includeHeader:
                metaDat = metaDat[
                    [
                        "capture_time",
                        "record_time",
                        "capture_id",
                        "record_id",
                        "capture_starttime",
                        "serialnumber",
                        "configuration",
                        "hostname",
                        "gitTag",
                        "gitBranch",
                        "nMovingPixel",
                    ]
                ]
            else:
                metaDat = metaDat[
                    [
                        "capture_time",
                        "record_time",
                        "capture_id",
                        "record_id",
                        "nMovingPixel",
                    ]
                ]
        else:
            raise ValueError(f"unknown asciiVersion {asciiVersion}")

        # else:
        #    metaDat = metaDat[['capture_time', 'record_time', 'capture_id', 'capture_starttime',
        #                       'serialnumber', 'configuration', 'hostname', 'gitTag', 'gitBranch', 'nMovingPixel']]

    # metaDat = metaDat.expand_dims({'record_starttime': [record_starttime]})

    metaDat["nThread"] = xr.zeros_like(metaDat["capture_id"])
    metaDat["nThread"].values[:] = nThread

    # save storage
    # metaDat["foundParticles"] = metaDat["foundParticles"].astype(np.uint32)
    # metaDat["movingObjects"] = metaDat["movingObjects"].astype(np.uint32)
    metaDat["nThread"] = metaDat["nThread"].astype(np.uint16)
    return metaDat


@tools.loopify_with_camera(endYesterday=False)
def createMetaFrames(
    case, camera, config, skipExisting=True, writeNc=True, doPlot=True
):
    """
    Create metadata frames for a given case and camera.

    This function processes all metadata files for a given case and camera
    to generate consolidated metadata frames for further processing.

    Parameters
    ----------
    case : str
        Case identifier
    camera : str
        Camera identifier
    config : dict or str
        Configuration dictionary or path to configuration file
    skipExisting : bool, optional
        Whether to skip existing files, default is True
    doPlot : bool, optional
        Do plot, default is True

    Returns
    -------
    xarray.Dataset or None
        Dataset with metadata or None if no data
    """

    # find files
    ff = files.FindFiles(case, camera, config)
    metaDat = None
    for fname0 in ff.listFiles("level0txt"):
        metaDat = createMetaFrames1(
            fname0, camera, config, skipExisting=skipExisting, writeNc=writeNc
        )

    if doPlot:
        fOut, fig = quicklooks.metaFramesQuicklook(
            case, camera, config, skipExisting=skipExisting
        )

    return metaDat  # only for testing


def createMetaFrames1(fname0, camera, config, skipExisting=True, writeNc=True):
    if config["nThreads"] is None:
        nThreads = 1
    else:
        nThreads = config["nThreads"]

    fn = files.Filenames(fname0, config)
    ff = files.FindFiles(fn.case, camera, config)
    fname0all = list(fn.fnameTxtAllThreads.values())

    if os.path.isfile(fn.fname.metaFrames) and skipExisting:
        log.info("%s exists" % fn.fname.metaFrames)
        return None

    if os.path.isfile(f"{fn.fname.metaFrames}.nodata") and skipExisting:
        log.info("%s.nodata exists" % fn.fname.metaFrames)
        return None

    if os.path.getsize(fname0.replace(config.movieExtension, "txt")) == 0:
        log.error("%s has size 0!" % fname0)
        with tools.open2(fn.fname.metaFrames + ".nodata", config, "w") as f:
            f.write("%s has size 0!" % fname0)
        return None

    # sometimes one thread file is missing. carefully check whether it migth be stuck in transfer
    tooFewThreads = len(fname0all) < int(nThreads)
    # presence of an event file is a good sign that at least some data has been transferred
    nextDayAvailable = os.path.isfile(ff.tomorrowObject.fnamesDaily.metaEvents)
    campaignEnded = config.end != "today"
    if tooFewThreads and (nextDayAvailable or campaignEnded):
        log.error("%s file of second thread missing!" % fname0)
        with tools.open2(fn.fname.metaFrames + ".nodata", config, "w") as f:
            f.write("%s file of second thread missing!" % fname0)
        return None

    metaDat, droppedFrames, beyondRepair = getMetaData(
        fname0all, camera, config, idOffset=0
    )

    if beyondRepair:
        print(
            f"{os.path.basename(fname0)}, broken beyond repair, {droppedFrames}, frames dropped\n"
        )

    if metaDat is not None:
        metaDat = tools.finishNc(metaDat, config.site, config.visssGen)
        if writeNc:
            tools.to_netcdf2(metaDat, config, fn.fname.metaFrames)
    else:
        with tools.open2(fn.fname.metaFrames + ".nodata", config, "w") as f:
            f.write("no data recorded")

    return metaDat


def getEvents(fnames0, config, fname0status=None):
    """
    Get event metadata for a given set of files.

    This function processes metadata files to extract event information
    including file start times, capture times, and camera status.

    Parameters
    ----------
    fnames0 : list
        List of metadata file names
    config : dict
        Configuration dictionary
    fname0status : str, optional
        Status file name, default is None

    Returns
    -------
    xarray.Dataset
        Dataset containing event metadata
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    metaDats = list()
    bins = [0] + list(range(11, 255, 10))
    bins4xr = list(range(10, 260, 10))
    for fname0Txt in fnames0:
        assert "txt" in fname0Txt
        fname0Img = fname0Txt.replace("txt", "jpg")
        fname0 = fname0Txt.replace("txt", config["movieExtension"])

        metaDat = {}
        res = _readHeaderData(fname0Txt, returnLasttime=True)

        (
            record_starttime,
            asciiVersion,
            gitTag,
            gitBranch,
            capture_starttime,
            capture_firsttime,
            capture_lasttime,
            last_id,
            serialnumber,
            configuration,
            hostname,
            cameraTemperature,
            transferQueueCurrentBlockCount,
            transferMaxBlockSize,
            ptpStatus,
        ) = res

        record_starttime = np.datetime64(record_starttime)

        if capture_starttime is None:
            # we need to keep trck of broken files because the number of "newfiles" and "brokenfiles" in the event file is
            # needed to keep track of completeness
            metaDat["event"] = xr.DataArray(
                ["brokenfile"], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["capture_starttime"] = xr.DataArray(
                np.array([np.nan]).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["capture_firsttime"] = xr.DataArray(
                np.array([np.nan]).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["capture_lasttime"] = xr.DataArray(
                np.array([np.nan]).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["serialnumber"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["configuration"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["hostname"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["gitTag"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["gitBranch"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["filename"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["ptpStatus"] = xr.DataArray(
                np.array([np.nan]).astype("object"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["cameraTemperature"] = xr.DataArray(
                [np.nan], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["transferQueueCurrentBlockCount"] = xr.DataArray(
                [-99],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["transferMaxBlockSize"] = xr.DataArray(
                [np.nan], dims=["file_starttime"], coords=[[record_starttime]]
            )
        else:
            capture_starttime = np.datetime64(capture_starttime)
            capture_firsttime = np.datetime64(capture_firsttime)
            capture_lasttime = np.datetime64(capture_lasttime)

            metaDat["event"] = xr.DataArray(
                ["newfile"], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["capture_starttime"] = xr.DataArray(
                np.array([capture_starttime], dtype="datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["capture_firsttime"] = xr.DataArray(
                np.array([capture_firsttime], dtype="datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["capture_lasttime"] = xr.DataArray(
                np.array([capture_lasttime], dtype="datetime64[ns]"),
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["serialnumber"] = xr.DataArray(
                [serialnumber], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["configuration"] = xr.DataArray(
                [configuration], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["hostname"] = xr.DataArray(
                [hostname], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["gitTag"] = xr.DataArray(
                [gitTag], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["gitBranch"] = xr.DataArray(
                [gitBranch], dims=["file_starttime"], coords=[[record_starttime]]
            )
            metaDat["filename"] = xr.DataArray(
                [fname0.split("/")[-1]],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["ptpStatus"] = xr.DataArray(
                [ptpStatus],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["cameraTemperature"] = xr.DataArray(
                [cameraTemperature],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["transferQueueCurrentBlockCount"] = xr.DataArray(
                [transferQueueCurrentBlockCount],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
            metaDat["transferMaxBlockSize"] = xr.DataArray(
                [transferMaxBlockSize],
                dims=["file_starttime"],
                coords=[[record_starttime]],
            )
        # estimate Blocking
        # using results of background estiamtor would be nice, but we don't have that inforation yet!
        img = plt.imread(fname0Img)[config.level1detect.height_offset :]
        nPixel = np.prod(img.shape)
        hist, _ = np.histogram(img.ravel(), bins=bins)
        hist = hist.cumsum() / nPixel
        metaDat["blocking"] = xr.DataArray(
            [hist],
            dims=["file_starttime", "blockingThreshold"],
            coords=[[record_starttime], bins4xr],
        )
        metaDat["brightnessMean"] = xr.DataArray(
            np.array([img.mean()]), dims=["file_starttime"], coords=[[record_starttime]]
        )
        metaDat["brightnessStd"] = xr.DataArray(
            np.array([img.std()]), dims=["file_starttime"], coords=[[record_starttime]]
        )

        metaDats.append(xr.Dataset(metaDat))

    if len(metaDats) > 0:
        metaDats = xr.concat(metaDats, dim="file_starttime")
    else:
        metaDats = {}
        metaDats["event"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["capture_starttime"] = xr.DataArray(
            np.array([], dtype="datetime64[ns]"), dims=["file_starttime"], coords=[[]]
        )
        metaDats["capture_firsttime"] = xr.DataArray(
            np.array([], dtype="datetime64[ns]"), dims=["file_starttime"], coords=[[]]
        )
        metaDats["capture_lasttime"] = xr.DataArray(
            np.array([], dtype="datetime64[ns]"), dims=["file_starttime"], coords=[[]]
        )
        metaDats["serialnumber"] = xr.DataArray(
            [], dims=["file_starttime"], coords=[[]]
        )
        metaDats["configuration"] = xr.DataArray(
            [], dims=["file_starttime"], coords=[[]]
        )
        metaDats["hostname"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["gitTag"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["gitBranch"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["filename"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["ptpStatus"] = xr.DataArray([], dims=["file_starttime"], coords=[[]])
        metaDats["cameraTemperature"] = xr.DataArray(
            [], dims=["file_starttime"], coords=[[]]
        )
        metaDats["transferQueueCurrentBlockCount"] = xr.DataArray(
            [], dims=["file_starttime"], coords=[[]]
        )
        metaDats["transferMaxBlockSize"] = xr.DataArray(
            [], dims=["file_starttime"], coords=[[]]
        )

        metaDats["blocking"] = xr.DataArray(
            np.zeros((0, len(bins4xr))),
            dims=["file_starttime", "blockingThreshold"],
            coords=[[], bins4xr],
        )
        metaDats["brightnessMean"] = xr.DataArray(
            np.array([]), dims=["file_starttime"], coords=[[]]
        )
        metaDats["brightnessStd"] = xr.DataArray(
            np.array([]), dims=["file_starttime"], coords=[[]]
        )
        metaDats = xr.Dataset(metaDats)

    # add status information
    if fname0status is not None:
        statusDat = pd.read_csv(
            fname0status,
            names=["file_starttime", "timestamp", "event", "user"],
            index_col=0,
        )
        # remove broken indices, don't know why statusDat.dropna() does not work directly
        statusDat = statusDat.loc[statusDat.index.dropna()]

        # more robust than pd.to_datetime
        statusDat.index = [np.datetime64(t) for t in statusDat.index]

        # remove faulty data
        statusDat = statusDat.iloc[~np.isnan(statusDat.index)]
        statusDat["event"] = [
            f"{e.lstrip()}-{u.lstrip()}"
            for e, u in zip(statusDat.event, statusDat.user)
        ]
        statusDat = statusDat["event"]
        statusDat = xr.Dataset({"event": xr.DataArray(statusDat)})
        statusDat = statusDat.rename(dim_0="file_starttime")
        try:
            metaDats = xr.merge(
                (metaDats, statusDat), join="outer", compat="no_conflicts"
            )

        except TypeError:
            metaDats = statusDat
            nEvents = len(statusDat["event"])
            metaDats["capture_starttime"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["capture_firsttime"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["capture_lasttime"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("datetime64[ns]"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["serialnumber"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["configuration"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["hostname"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["gitTag"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["gitBranch"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["filename"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )
            metaDats["ptpStatus"] = xr.DataArray(
                (np.zeros(nEvents) * np.nan).astype("object"),
                dims=["file_starttime"],
                coords=[metaDats.file_starttime],
            )

            metaDats["blocking"] = xr.DataArray(
                np.zeros((nEvents, len(bins4xr))) * np.nan,
                dims=["file_starttime", "blockingThreshold"],
                coords=[metaDats.file_starttime, bins4xr],
            )

    # no status files are available for MOSAiC, therefore use netcdf generated from log files to figure out when instrument was started (and clock was reset!)
    if (len(fnames0) > 0) and (config.site == "mosaic"):
        # becuase this is the only time we use _softwareStarttimes files, name is hardcoded here:
        path = config.pathOut.format(level="metaEvents", version=__version__)
        restartFile = f'{path}/{"_".join(fname0.split("/")[-1].split("_")[:-1])}_softwareStarttimes.nc'
        restartDat = xr.open_dataset(restartFile)
        case = fname0.split("/")[-1].split("_")[-1].split("-")[0]
        startTime = np.datetime64(f"{case[:4]}-{case[4:6]}-{case[6:8]}")
        endTime = startTime + np.timedelta64(1, "D")
        ttWindow = (restartDat.software_starttimes >= startTime) & (
            restartDat.software_starttimes < endTime
        )
        restartDat = restartDat.isel(software_starttimes=ttWindow).software_starttimes
        if len(restartDat) > 0:
            statusDat = xr.Dataset()
            statusDat["event"] = xr.DataArray(
                ["start-user"] * len(restartDat),
                coords=[restartDat],
                dims=["file_starttime"],
            )
            metaDats = xr.merge((metaDats, statusDat))

    # xarray bug: merge throws away dateime64 unit "ns" if len(metaDats) == 0 before merge, so make sure it is correct
    metaDats = metaDats.assign_coords(
        file_starttime=metaDats.file_starttime.astype("datetime64[ns]")
    )
    metaDats["capture_starttime"] = metaDats["capture_starttime"].astype(
        "datetime64[ns]"
    )
    metaDats["capture_firsttime"] = metaDats["capture_firsttime"].astype(
        "datetime64[ns]"
    )
    metaDats["capture_lasttime"] = metaDats["capture_lasttime"].astype("datetime64[ns]")

    return metaDats


@tools.loopify_with_camera(endYesterday=False)
def createEvent(
    case,
    camera,
    config,
    skipExisting=True,
    quiet=False,
    writeNc=True,
    version=__version__,
):
    """
    Create event file for a given case and camera.

    This function generates event metadata files that summarize the status
    and timing information for all video files in a given case.

    Parameters
    ----------
    case : str
        Case identifier
    camera : str
        Camera identifier
    config : dict or str
        Configuration dictionary or path to configuration file
    skipExisting : bool, optional
        Whether to skip existing files, default is True
    quiet : bool, optional
        Whether to suppress logging output, default is False
    version : str, optional
        Version string, default is __version__

    Returns
    -------
    xarray.Dataset or None
        Dataset with event metadata or None if skipped
    """
    # case is always daily for events!
    case = case.split("-")[0]
    if type(config) is str:
        config = tools.readSettings(config)

    if camera in ["leader", "follower"]:
        camera = config[camera]

    fn = files.FindFiles(case, camera, config, version)
    fnames0 = fn.listFiles("level0txt")

    fname0status = fn.listFiles(level="level0status")
    if len(fname0status) > 0:
        fname0status = fname0status[0]
    else:
        fname0status = None

    # ff = files.Filenames(fnames0[0], config, version)
    eventFile = fn.fnamesDaily.metaEvents

    if skipExisting and os.path.isfile(eventFile):
        with xr.open_dataset(eventFile) as eventDat:
            eventDat.load()
        if len(eventDat.data_vars) == 0:
            log.info("eventDat empty, redoing event file")

        # check whether status file is newer than event file, consider 6 hour buffer for data transfer
        elif (fname0status is not None) and (
            os.path.getmtime(eventFile) < (os.path.getmtime(fname0status) + 60 * 60 * 6)
        ):
            log.info("status file was recently updated, redoing event file")

        else:
            if "noLevel0Files" in eventDat.attrs:
                nFiles = eventDat.attrs["noLevel0Files"]
            else:
                nFiles = sum(eventDat.event == "newfile") + sum(
                    eventDat.event == "brokenfile"
                )
                nFiles = int(nFiles.values)

            if nFiles == len(fnames0):
                if not quiet:
                    log.info(tools.concat("Skipping", case, eventFile))

                return None
            else:
                log.info(
                    tools.concat(
                        "redoing event file, so far we had only",
                        nFiles,
                        "of",
                        len(fnames0),
                        "files",
                    )
                )

    log.info(tools.concat("Running", case, eventFile))
    metaDats = getEvents(fnames0, config, fname0status=fname0status)
    try:
        assert len(metaDats.file_starttime) > 0
        metaDats = tools.finishNc(metaDats, config.site, config.visssGen)
        nFiles = sum(metaDats.event == "newfile") + sum(metaDats.event == "brokenfile")
        nFiles = int(nFiles.values)
        metaDats.attrs["noLevel0Files"] = nFiles
        if writeNc:
            tools.to_netcdf2(metaDats, config, eventFile)

    except (ValueError, AssertionError):
        print("NO DATA", case, eventFile)

    return metaDats
